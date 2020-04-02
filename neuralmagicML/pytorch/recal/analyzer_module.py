"""
Code related to monitoring, analyzing, and reporting info for Modules in PyTorch.
Records things like FLOPS, input and output shapes, kernel shapes, etc.
"""

from typing import Union, Tuple, List
import numpy

from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    Softmax,
    Softmax2d,
    Threshold,
    ReLU,
    ReLU6,
    RReLU,
    LeakyReLU,
    PReLU,
    ELU,
    CELU,
    SELU,
    GLU,
    Hardtanh,
    Tanh,
    Sigmoid,
    LogSigmoid,
)
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd
from torch.nn.modules.pooling import _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd
from torch.utils.hooks import RemovableHandle

from neuralmagicML.recal import AnalyzedLayerDesc
from neuralmagicML.pytorch.utils import get_layer, get_prunable_layers


__all__ = ["ModuleAnalyzer"]


class ModuleAnalyzer(object):
    """
    An analyzer implementation for monitoring the execution profile and graph of
    a Module in PyTorch.

    :param module: the module to analyze
    :param enabled: True to enable the hooks for analyzing and actively track,
        False to disable and not track
    """

    def __init__(self, module: Module, enabled: bool = False):
        super(ModuleAnalyzer, self).__init__()
        self._module = module
        self._hooks = None  # type: List[RemovableHandle]
        self._forward_called = False
        self._enabled = False
        self._call_count = -1
        self.enabled = enabled

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
            self._param_grad = None
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

        for (name, _) in get_prunable_layers(self._module):
            desc = self.layer_desc(name)
            descs.append(desc)

        descs.sort(key=lambda val: val.execution_order)

        return descs

    def _create_hooks(self):
        self._delete_hooks()
        self._forward_called = False
        self._call_count = -1
        self._hooks = []
        self._hooks.append(self._create_hook(self._module, None))

        for name, mod in self._module.named_modules():
            self._hooks.append(self._create_hook(mod, name))

    def _delete_hooks(self):
        if self._hooks is not None:
            for hook in self._hooks:
                hook.remove()

            self._hooks.clear()

    def _create_hook(self, mod: Module, name: str) -> RemovableHandle:
        mod._analyzed_layer_desc = None
        mod._analyzed_layer_name = name

        if isinstance(mod, _ConvNd):
            return mod.register_forward_hook(self._conv_hook)

        if isinstance(mod, Linear):
            return mod.register_forward_hook(self._linear_hook)

        if isinstance(mod, _BatchNorm):
            return mod.register_forward_hook(self._bn_hook)

        if isinstance(mod, _MaxPoolNd) or isinstance(mod, _AvgPoolNd):
            return mod.register_forward_hook(self._pool_hook)

        if isinstance(mod, _AdaptiveAvgPoolNd) or isinstance(mod, _AdaptiveMaxPoolNd):
            return mod.register_forward_hook(self._adaptive_pool_hook)

        if (
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
            return mod.register_forward_hook(self._activation_hook)

        if isinstance(mod, Softmax) or isinstance(mod, Softmax2d):
            return mod.register_forward_hook(self._softmax_hook)

        return mod.register_forward_hook(self._module_hook)

    def _module_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "any", inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

    def _conv_hook(
        self,
        mod: _ConvNd,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "conv", inp, out)

        params = (
            {"weight": mod.weight}
            if mod.bias is None
            else {"weight": mod.weight, "bias": mod.bias}
        )
        prunable_params = {"weight": mod.weight}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        mult_per_out_pix = float(numpy.prod(mod.kernel_size)) * mod.in_channels
        add_per_out_pix = 1 if mod.bias is not None else 0
        out_pix = float(numpy.prod(out[0].shape[1:]))

        # total flops counts the cost of summing the multiplications together activation well
        # most implementations and papers do not include this cost
        desc.flops = (mult_per_out_pix + add_per_out_pix) * out_pix
        desc.total_flops = (mult_per_out_pix * 2 + add_per_out_pix) * out_pix

    def _linear_hook(
        self,
        mod: Linear,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "linear", inp, out)

        params = (
            {"weight": mod.weight}
            if mod.bias is None
            else {"weight": mod.weight, "bias": mod.bias}
        )
        prunable_params = {"weight": mod.weight}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        mult_per_out_pix = mod.in_features
        add_per_out_pix = 1 if mod.bias is not None else 0
        out_pix = float(numpy.prod(out[0].shape[1:]))

        # total flops counts the cost of summing the multiplications together activation well
        # most implementations and papers do not include this cost
        desc.flops = (mult_per_out_pix + add_per_out_pix) * out_pix
        desc.total_flops = (mult_per_out_pix * 2 + add_per_out_pix) * out_pix

    def _bn_hook(
        self,
        mod: Linear,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "batch_norm", inp, out)

        params = (
            {"weight": mod.weight}
            if mod.bias is None
            else {"weight": mod.weight, "bias": mod.bias}
        )
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        # 4 elementwise operations on the output space, just need to add all of them up
        desc.flops = 4 * float(numpy.prod(out[0].shape[1:]))
        desc.total_flops = desc.flops

    def _pool_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "pool", inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        flops_per_out_pix = float(numpy.prod(mod.kernel_size) + 1)
        out_pix = float(numpy.prod(out[0].shape[1:]))

        desc.flops = flops_per_out_pix * out_pix
        desc.total_flops = desc.flops

    def _adaptive_pool_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "global_pool", inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        stride = tuple(
            inp[0].shape[i] // out[0].shape[i] for i in range(2, len(inp[0].shape))
        )
        kernel_size = tuple(
            inp[0].shape[i] - (out[0].shape[i] - 1) * stride[i - 2]
            for i in range(2, len(inp[0].shape))
        )
        flops_per_out_pix = float(numpy.prod(kernel_size))
        out_pix = float(numpy.prod(out[0].shape[1:]))

        desc.flops = flops_per_out_pix * out_pix
        desc.total_flops = desc.flops

    def _activation_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "act", inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        # making assumption that flops spent is one per element (so swish is counted the same activation ReLU)
        desc.flops = float(numpy.prod(out[0].shape[1:]))
        desc.total_flops = desc.flops

    def _softmax_hook(
        self,
        mod: Union[_MaxPoolNd, _AvgPoolNd],
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ):
        desc, inp, out = self._init_hook(mod, "softmax", inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum(
            [(val == 0).sum().item() for val in prunable_params.values()]
        )
        desc.params_dims = {
            key: tuple(s for s in val.shape) for key, val in params.items()
        }
        desc.prunable_params_dims = {
            key: tuple(s for s in val.shape) for key, val in prunable_params.items()
        }

        flops_per_channel = (
            2 if len(out[0].shape) < 3 else float(numpy.prod(out[0].shape[2:]))
        )
        desc.flops = flops_per_channel * out[0].shape[1]
        desc.total_flops = desc.flops

    def _init_hook(
        self,
        mod: Module,
        type_: str,
        inp: Union[Tuple[Tensor, ...], Tensor],
        out: Union[Tuple[Tensor, ...], Tensor],
    ) -> Tuple[AnalyzedLayerDesc, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        self._forward_called = True
        self._call_count += 1

        if isinstance(inp, Tensor):
            inp = (inp,)

        if isinstance(out, Tensor):
            out = (out,)

        mod._analyzed_layer_desc = AnalyzedLayerDesc(
            name=mod._analyzed_layer_name,
            type_=type_,
            execution_order=self._call_count,
            input_shape=tuple(
                tuple(ii for ii in i.shape) for i in inp if isinstance(i, Tensor)
            ),
            output_shape=tuple(
                tuple(oo for oo in o.shape) for o in out if isinstance(o, Tensor)
            ),
        )

        return mod._analyzed_layer_desc, inp, out

    @staticmethod
    def _mod_desc(mod: Module) -> AnalyzedLayerDesc:
        children = []
        for _, child in mod.named_modules():
            if child != mod:
                children.append(child)

        merge_descs = [
            ModuleAnalyzer._mod_desc(child) for child in children
        ]  # type: List[AnalyzedLayerDesc]

        return AnalyzedLayerDesc.merge_descs(mod._analyzed_layer_desc, merge_descs)
