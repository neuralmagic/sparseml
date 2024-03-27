# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for overall sparsity and forward  FLOPs (floating-point operations)
estimation for neural networks.
"""

import numbers
from typing import List, Tuple, Union

import numpy
from torch import Tensor
from torch.nn import (
    CELU,
    ELU,
    GLU,
    SELU,
    Hardtanh,
    LeakyReLU,
    Linear,
    LogSigmoid,
    Module,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    Sigmoid,
    Softmax,
    Softmax2d,
    Tanh,
    Threshold,
)
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import (
    _AdaptiveAvgPoolNd,
    _AdaptiveMaxPoolNd,
    _AvgPoolNd,
    _MaxPoolNd,
)
from torch.utils.hooks import RemovableHandle

from sparseml.optim import AnalyzedLayerDesc
from sparseml.pytorch.utils import get_layer, get_prunable_layers


__all__ = ["ModuleAnalyzer"]


class ModuleAnalyzer(object):
    """
    An analyzer implementation for monitoring the execution profile and graph of
    a Module in PyTorch.

    :param module: the module to analyze
    :param enabled: True to enable the hooks for analyzing and actively track,
        False to disable and not track
    :param ignore_zero: whether zeros should be excluded from FLOPs (standard
        when estimating 'theoretical' FLOPs in sparse networks
    : param
    :param multiply_adds: Whether total flops includes the cost of summing the
        multiplications together
    """

    def __init__(
        self,
        module: Module,
        enabled: bool = False,
        ignore_zero=True,
        multiply_adds=True,
    ):
        super(ModuleAnalyzer, self).__init__()
        self._module = module
        self._hooks = None  # type: List[RemovableHandle]
        self._forward_called = False
        self._enabled = False
        self._call_count = -1
        self.enabled = enabled
        self._ignore_zero = ignore_zero
        self._multiply_adds = multiply_adds

    def __del__(self):
        self._delete_hooks()

    @property
    def enabled(self) -> bool:
        """
        :return: True if enabled and the hooks for analyzing are active, False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to enable the hooks for analyzing, False to disable
        """
        if value and not self._enabled:
            self._create_hooks()
            self._params_grad = None
        elif not value and self._enabled:
            self._delete_hooks()

        self._enabled = value

    @property
    def module(self) -> Module:
        """
        :return: The module that is being actively analyzed
        """
        return self._module

    def layer_desc(self, name: Union[str, None] = None) -> AnalyzedLayerDesc:
        """
        Get a specific layer's description within the Module.
        Set to None to get the overall Module's description.

        :param name: name of the layer to get a description for,
            None for an overall description
        :return: the analyzed layer description for the given name
        """
        if not self._forward_called:
            raise RuntimeError(
                "module must have forward called with sample input "
                "before getting a layer desc"
            )

        mod = get_layer(name, self._module) if name is not None else self._module

        return ModuleAnalyzer._mod_desc(mod)

    def ks_layer_descs(self) -> List[AnalyzedLayerDesc]:
        """
        Get the descriptions for all layers in the module that support kernel sparsity
        (model pruning). Ex: all convolutions and linear layers.

        :return: a list of descriptions for all layers in the module that support ks
        """
        descs = []

        for name, _ in get_prunable_layers(self._module):
            desc = self.layer_desc(name)

            if desc is None:
                print("analyzer: no description found for {}".format(name))
            else:
                descs.append(desc)

        descs.sort(key=lambda val: val.execution_order)

        return descs

    def _create_hooks(self):
        self._delete_hooks()
        self._forward_called = False
        self._call_count = -1
        self._hooks = []

        for name, mod in self._module.named_modules():
            self._hooks.extend(
                self._create_mod_hooks(mod, name if mod != self._module else None)
            )

    def _delete_hooks(self):
        if self._hooks is not None:
            for hook in self._hooks:
                hook.remove()

            self._hooks.clear()

    def _create_mod_hooks(self, mod: Module, name: str) -> List[RemovableHandle]:
        mod._analyzed_layer_desc = None
        mod._analyzed_layer_name = name

        forward_pre_hook = mod.register_forward_pre_hook(self._forward_pre_hook)

        if isinstance(mod, _ConvNd):
            forward_hook = mod.register_forward_hook(self._conv_hook)
        elif isinstance(mod, Linear):
            forward_hook = mod.register_forward_hook(self._linear_hook)
        elif isinstance(mod, _BatchNorm):
            forward_hook = mod.register_forward_hook(self._bn_hook)
        elif isinstance(mod, _MaxPoolNd) or isinstance(mod, _AvgPoolNd):
            forward_hook = mod.register_forward_hook(self._pool_hook)
        elif isinstance(mod, _AdaptiveAvgPoolNd) or isinstance(mod, _AdaptiveMaxPoolNd):
            forward_hook = mod.register_forward_hook(self._adaptive_pool_hook)
        elif (
            isinstance(mod, Threshold)
            or isinstance(mod, ReLU)
            or isinstance(mod, ReLU6)
            or isinstance(mod, RReLU)
            or isinstance(mod, LeakyReLU)
            or isinstance(mod, PReLU)
            or isinstance(mod, ELU)
            or isinstance(mod, CELU)
            or isinstance(mod, SELU)
            or isinstance(mod, GLU)
            or isinstance(mod, Hardtanh)
            or isinstance(mod, Tanh)
            or isinstance(mod, Sigmoid)
            or isinstance(mod, LogSigmoid)
        ):
            forward_hook = mod.register_forward_hook(self._activation_hook)
        elif isinstance(mod, Softmax) or isinstance(mod, Softmax2d):
            forward_hook = mod.register_forward_hook(self._softmax_hook)
        else:
            forward_hook = mod.register_forward_hook(self._module_hook)

        return [forward_pre_hook, forward_hook]

    def _forward_pre_hook(
        self,
        mod: Module,
        inp: Union[Tuple[Tensor, ...], Tensor],
    ):
        self._call_count += 1

        if mod._analyzed_layer_desc is not None:
            return

        mod._analyzed_layer_desc = AnalyzedLayerDesc(
            name=mod._analyzed_layer_name,
            type_=mod.__class__.__name__,
            execution_order=self._call_count,
            flops=0,
            total_flops=0,
        )

    def _init_forward_hook(
        self,
        mod: Module,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ) -> Tuple[AnalyzedLayerDesc, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        self._forward_called = True

        if isinstance(inp, Tensor):
            inp = (inp,)

        if isinstance(out, Tensor):
            out = (out,)

        desc = mod._analyzed_layer_desc
        desc.input_shape = tuple(
            tuple(ii for ii in i.shape) for i in inp if isinstance(i, Tensor)
        )
        desc.output_shape = tuple(
            tuple(oo for oo in o.shape) for o in out if isinstance(o, Tensor)
        )

        return desc, inp, out

    def _module_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

    def _conv_hook(
        self,
        mod: _ConvNd,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        desc.params = mod.weight.data.numel() + (
            mod.bias.data.numel() if mod.bias is not None else 0
        )
        desc.prunable_params = mod.weight.data.numel()
        desc.zeroed_params = desc.prunable_params - mod.weight.data.count_nonzero()

        batch_size, input_channels, input_height, input_width = inp[0].size()
        _, output_channels, output_height, output_width = out[0].size()

        bias_ops = 1 if mod.bias is not None else 0

        num_weight_params = (
            (mod.weight.data != 0.0).float().sum()
            if self._ignore_zero
            else mod.weight.data.nelement()
        )

        flops = (
            (
                num_weight_params * (2 if self._multiply_adds else 1)
                + bias_ops * output_channels
            )
            * output_height
            * output_width
            * batch_size
        )

        desc.flops = flops
        desc.total_flops += desc.flops

    def _linear_hook(
        self,
        mod: Linear,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        desc.params = mod.weight.data.numel() + (
            mod.bias.data.numel() if mod.bias is not None else 0
        )
        desc.prunable_params = mod.weight.data.numel()
        desc.zeroed_params = desc.prunable_params - mod.weight.data.count_nonzero()

        batch_size = inp[0].size(0) if inp[0].dim() == 2 else 1

        num_weight_params = (
            (mod.weight.data != 0.0).float().sum()
            if self._ignore_zero
            else mod.weight.data.nelement()
        )
        weight_ops = num_weight_params * (2 if self._multiply_adds else 1)
        bias_ops = mod.bias.nelement() if mod.bias is not None else 0

        desc.flops = batch_size * (weight_ops + bias_ops)
        desc.total_flops += desc.flops

    def _bn_hook(
        self,
        mod: Linear,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        desc.params = mod.weight.data.numel() + (
            mod.bias.data.numel() if mod.bias is not None else 0
        )
        desc.prunable_params = mod.weight.data.numel()
        desc.zeroed_params = desc.prunable_params - mod.weight.data.count_nonzero()

        desc.flops = 2 * float(inp[0].nelement())
        desc.total_flops += desc.flops

    def _pool_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = 0
        desc.zeroed_params = 0

        batch_size, input_channels, input_height, input_width = inp[0].size()
        batch_size, output_channels, output_height, output_width = out[0].size()

        if isinstance(mod.kernel_size, numbers.Number) or mod.kernel_size.dim() == 1:
            kernel_ops = mod.kernel_size * mod.kernel_size
        else:
            kernel_ops = numpy.prod(mod.kernel_size)
        flops = kernel_ops * output_channels * output_height * output_width * batch_size

        desc.flops = flops
        desc.total_flops += desc.flops

    def _adaptive_pool_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        params = {key: val for key, val in mod.named_parameters()}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = 0
        desc.zeroed_params = 0

        stride = tuple(
            inp[0].shape[i] // out[0].shape[i] for i in range(2, len(inp[0].shape))
        )
        kernel_size = tuple(
            inp[0].shape[i] - (out[0].shape[i] - 1) * stride[i - 2]
            for i in range(2, len(inp[0].shape))
        )
        kernel_ops = numpy.prod(kernel_size)

        batch_size, output_channels, output_height, output_width = out[0].size()

        flops = kernel_ops * output_channels * output_height * output_width * batch_size

        desc.flops = flops
        desc.total_flops += desc.flops

    def _activation_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        params = {key: val for key, val in mod.named_parameters()}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = 0
        desc.zeroed_params = 0

        # making assumption that flops spent is one per element
        # (so swish is counted the same activation ReLU)
        # FIXME (can't really be fixed). Some standard architectures,
        # such as a standard ResNet use the same activation (ReLU) object
        # for all of the places that it appears in the net, which works
        # fine because it's stateless. But it makes it hard to count per-
        # batch forward FLOPs correctly, since a single forward pass
        # through the network is actually multiple passes trhough the
        # activation. So the per-batch FLOPs are undercounted (slightly,
        # since activations are very few FLOPs in general), but total
        # (cumulative) FLOPs are counted correctly.
        desc.flops = float(inp[0].nelement())
        desc.total_flops += desc.flops

    def _softmax_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_forward_hook(mod, inp, out)

        params = {key: val for key, val in mod.named_parameters()}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = 0
        desc.zeroed_params = 0

        flops_per_channel = (
            2 if len(out[0].shape) < 3 else float(numpy.prod(out[0].shape[2:]))
        )
        desc.flops = flops_per_channel * out[0].shape[1]
        desc.total_flops += desc.flops

    @staticmethod
    def _mod_desc(mod: Module) -> AnalyzedLayerDesc:
        child_descs = []
        for _, child in mod.named_children():
            if child != mod:
                child_desc = ModuleAnalyzer._mod_desc(child)

                if child_desc:
                    child_descs.append(child_desc)

        if not mod._analyzed_layer_desc:
            return None

        return AnalyzedLayerDesc.merge_descs(mod._analyzed_layer_desc, child_descs)
