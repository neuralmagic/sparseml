from typing import Union, Tuple, List
import numpy
import json
from torch import Tensor, Size
from torch.nn import (
    Module, Linear, Softmax, Softmax2d,
    Threshold, ReLU, ReLU6, RReLU, LeakyReLU, PReLU, ELU, CELU, SELU, GLU,
    Hardtanh, Tanh, Sigmoid, LogSigmoid
)
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd
from torch.nn.modules.pooling import _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd
from torch.utils.hooks import RemovableHandle

from .utils import get_layer


__all__ = ['ModuleAnalyzer', 'AnalyzedLayerDesc']


class AnalyzedLayerDesc(object):
    def __init__(self, name: str, call_order: int = -1, input_shape: Tuple[Size, ...] = None,
                 output_shape: Tuple[Size, ...] = None, kernel_size: Tuple[int, ...] = None):
        self._name = name
        self._call_order = call_order

        self._input_shape = None
        self._output_shape = None
        self._in_channels = -1
        self._out_channels = -1
        self.input_shape = input_shape
        self.output_shape = output_shape

        self._kernel_size = kernel_size
        self._params = 0
        self._flops = 0
        self._total_flops = 0

    def __repr__(self):
        params = {
            'call_order': str(self.call_order),
            'input_shape': ','.join([str(i) for i in self.input_shape]) if self.input_shape is not None else None,
            'output_shape': ','.join([str(i) for i in self.output_shape]) if self.output_shape is not None else None,
            'in_channels': str(self.in_channels),
            'out_channels': str(self.out_channels),
            'kernel_size': str(self.kernel_size),
            'params': str(self.params),
            'flops': str(self.flops),
            'total_flops': str(self.total_flops)
        }

        return 'AnalyzedLayerDesc({})'.format(json.dumps(params))

    @property
    def name(self) -> str:
        return self._name

    @property
    def call_order(self) -> int:
        return self._call_order

    @call_order.setter
    def call_order(self, value: int):
        self._call_order = value

    @property
    def input_shape(self) -> Tuple[Size, ...]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Tuple[Size, ...]):
        self._input_shape = value
        self._in_channels = value[0][1] if value is not None else -1

    @property
    def output_shape(self) -> Tuple[Size, ...]:
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value: Tuple[Size, ...]):
        self._output_shape = value
        self._out_channels = value[0][1] if value is not None else -1

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel_size(self) -> Tuple[int]:
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value: Tuple[int]):
        self._kernel_size = value

    @property
    def params(self) -> int:
        return self._params

    @params.setter
    def params(self, value: int):
        self._params = value

    @property
    def flops(self) -> int:
        return self._flops

    @flops.setter
    def flops(self, value: int):
        self._flops = value

    @property
    def total_flops(self) -> int:
        return self._total_flops

    @total_flops.setter
    def total_flops(self, value: int):
        self._total_flops = value


class ModuleAnalyzer(object):
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
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value and not self._enabled:
            self._create_hooks()
            self._param_grad = None
        elif not value and self._enabled:
            self._delete_hooks()

        self._enabled = value

    def layer_desc(self, name: Union[str, None] = None) -> AnalyzedLayerDesc:
        if not self._forward_called:
            raise RuntimeError('module must have forward called with sample input before getting a layer desc')

        mod = get_layer(name, self._module) if name is not None else self._module

        return ModuleAnalyzer._mod_desc(mod)

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

        if (isinstance(mod, Threshold) or isinstance(mod, ReLU) or isinstance(mod, ReLU6) or isinstance(mod, RReLU) or
                isinstance(mod, LeakyReLU) or isinstance(mod, PReLU) or isinstance(mod, ELU) or isinstance(mod, CELU) or
                isinstance(mod, SELU) or isinstance(mod, GLU) or isinstance(mod, Hardtanh) or isinstance(mod, Tanh) or
                isinstance(mod, Sigmoid) or isinstance(mod, LogSigmoid)):
            return mod.register_forward_hook(self._activation_hook)

        if isinstance(mod, Softmax) or isinstance(mod, Softmax2d):
            return mod.register_forward_hook(self._softmax_hook)

        return mod.register_forward_hook(self._module_hook)

    def _module_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                     out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

    def _conv_hook(self, mod: _ConvNd, inp: Union[Tuple[Tensor, ...], Tensor], out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.kernel_size = tuple(k for k in mod.kernel_size)
        desc.params = mod.weight.numel() if mod.bias is None else mod.weight.numel() + mod.bias.numel()

        mult_per_out_pix = numpy.prod(mod.kernel_size) * mod.in_channels
        add_per_out_pix = 1 if mod.bias is not None else 0
        out_pix = numpy.prod(out[0].shape[1:])

        # total flops counts the cost of summing the multiplications together activation well
        # most implementations and papers do not include this cost
        desc.flops = (mult_per_out_pix + add_per_out_pix) * out_pix
        desc.total_flops = (mult_per_out_pix * 2 + add_per_out_pix) * out_pix

    def _linear_hook(self, mod: Linear, inp: Union[Tuple[Tensor, ...], Tensor], out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.kernel_size = (1,)
        desc.params = mod.weight.numel() if mod.bias is None else mod.weight.numel() + mod.bias.numel()

        mult_per_out_pix = mod.in_features
        add_per_out_pix = 1 if mod.bias is not None else 0
        out_pix = numpy.prod(out[0].shape[1:])

        # total flops counts the cost of summing the multiplications together activation well
        # most implementations and papers do not include this cost
        desc.flops = (mult_per_out_pix + add_per_out_pix) * out_pix
        desc.total_flops = (mult_per_out_pix * 2 + add_per_out_pix) * out_pix

    def _bn_hook(self, mod: Linear, inp: Union[Tuple[Tensor, ...], Tensor], out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.params = mod.weight.numel() if mod.bias is None else mod.weight.numel() + mod.bias.numel()

        # 4 elementwise operations on the output space, just need to add all of them up
        desc.flops = 4 * numpy.prod(out[0].shape[1:])
        desc.total_flops = desc.flops

    def _pool_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                   out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

        flops_per_out_pix = numpy.prod(mod.kernel_size) + 1
        out_pix = numpy.prod(out[0].shape[1:])

        desc.flops = flops_per_out_pix * out_pix
        desc.total_flops = desc.flops

    def _adaptive_pool_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                            out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

        stride = tuple(inp[0].shape[i] // out[0].shape[i] for i in range(2, len(inp[0].shape)))
        kernel_size = tuple(inp[0].shape[i] - (out[0].shape[i] - 1) * stride[i - 2]
                            for i in range(2, len(inp[0].shape)))
        flops_per_out_pix = numpy.prod(kernel_size)
        out_pix = numpy.prod(out[0].shape[1:])

        desc.flops = flops_per_out_pix * out_pix
        desc.total_flops = desc.flops

    def _activation_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                         out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

        # making assumption that flops spent is one per element (so swish is counted the same activation ReLU)
        desc.flops = numpy.prod(out[0].shape[1:])
        desc.total_flops = desc.flops

    def _softmax_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                      out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

        flops_per_channel = 2 if len(out[0].shape) < 3 else numpy.prod(out[0].shape[2:])
        desc.flops = flops_per_channel * out[0].shape[1]
        desc.total_flops = desc.flops

    def _init_hook(self, mod: Module, inp: Union[Tuple[Tensor, ...], Tensor],
                   out: Union[Tuple[Tensor, ...], Tensor]) -> Tuple[AnalyzedLayerDesc, Tuple[Tensor, ...],
                                                                    Tuple[Tensor, ...]]:
        self._forward_called = True
        self._call_count += 1

        if isinstance(inp, Tensor):
            inp = (inp,)

        if isinstance(out, Tensor):
            out = (out,)

        desc = AnalyzedLayerDesc(name=mod._analyzed_layer_name,
                                 call_order=self._call_count,
                                 input_shape=tuple(i.shape for i in inp if isinstance(i, Tensor)),
                                 output_shape=tuple(o.shape for o in out if isinstance(o, Tensor)))
        mod._analyzed_layer_desc = desc

        return desc, inp, out

    @staticmethod
    def _mod_desc(mod: Module) -> AnalyzedLayerDesc:
        children = []
        for _, child in mod.named_modules():
            if child != mod:
                children.append(child)

        merge_descs = [mod._analyzed_layer_desc] if len(children) < 1 else \
            [ModuleAnalyzer._mod_desc(child) for child in children]  # type: List[AnalyzedLayerDesc]
        desc = AnalyzedLayerDesc(name=mod._analyzed_layer_desc.name,
                                 call_order=mod._analyzed_layer_desc.call_order,
                                 input_shape=mod._analyzed_layer_desc.input_shape,
                                 output_shape=mod._analyzed_layer_desc.output_shape,
                                 kernel_size=mod._analyzed_layer_desc.kernel_size)

        for merge in merge_descs:
            desc.flops += merge.flops
            desc.total_flops += merge.total_flops
            desc.params += merge.params

        return desc
