from typing import Union, Tuple, List, Dict, Any
import numpy
import json

from torch import Tensor
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

from ..utils import clean_path, create_parent_dirs, get_layer, get_conv_layers, get_linear_layers


__all__ = ['ModuleAnalyzer', 'AnalyzedLayerDesc',
           'model_ks_desc', 'save_model_ks_desc']


class AnalyzedLayerDesc(object):
    def __init__(self, name: str, type_: str,
                 params: int = 0, zeroed_params: int = 0, prunable_params: int = 0,
                 params_dims: Dict[str, Tuple[int, ...]] = None,
                 prunable_params_dims: Dict[str, Tuple[int, ...]] = None,
                 execution_order: int = -1,
                 input_shape: Tuple[Tuple[int, ...], ...] = None, output_shape: Tuple[Tuple[int, ...], ...] = None,
                 flops: int = 0, total_flops: int = 0):
        self.name = name
        self.type_ = type_

        self.params = params
        self.prunable_params = prunable_params
        self.zeroed_params = zeroed_params
        self.params_dims = params_dims
        self.prunable_params_dims = prunable_params_dims

        self.execution_order = execution_order
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.flops = flops
        self.total_flops = total_flops

    def __repr__(self):
        return 'AnalyzedLayerDesc({})'.format(self.json())

    @property
    def terminal(self) -> bool:
        return self.params_dims is not None

    @property
    def prunable(self) -> bool:
        return self.prunable_params > 0

    def dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type_,
            'params': self.params,
            'zeroed_params': self.zeroed_params,
            'prunable_params': self.prunable_params,
            'params_dims': self.params_dims,
            'prunable_params_dims': self.prunable_params_dims,
            'execution_order': self.execution_order,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'flops': self.flops,
            'total_flops': self.total_flops,
            'terminal': self.terminal,
            'prunable': self.prunable
        }

    def json(self) -> str:
        return json.dumps(self.dict())

    @staticmethod
    def merge_descs(orig, descs):
        merged = AnalyzedLayerDesc(
            name=orig.name, type_=orig.type_,
            params=orig.params, zeroed_params=orig.zeroed_params, prunable_params=orig.prunable_params,
            params_dims=orig.params_dims, prunable_params_dims=orig.prunable_params_dims,
            execution_order=orig.execution_order,
            input_shape=orig.input_shape, output_shape=orig.output_shape,
            flops=orig.flops, total_flops=orig.total_flops
        )

        for desc in descs:
            merged.flops += desc.flops
            merged.total_flops += desc.total_flops
            merged.params += desc.params
            merged.prunable_params += desc.prunable_params
            merged.zeroed_params += desc.zeroed_params

        return merged


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

    @property
    def module(self) -> Module:
        return self._module

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
        desc, inp, out = self._init_hook(mod, 'any', inp, out)
        desc.params = sum(param.numel() for param in mod.parameters())

    def _conv_hook(self, mod: _ConvNd, inp: Union[Tuple[Tensor, ...], Tensor], out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'conv', inp, out)

        params = {'weight': mod.weight} if mod.bias is None else {'weight': mod.weight, 'bias': mod.bias}
        prunable_params = {'weight': mod.weight}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        mult_per_out_pix = float(numpy.prod(mod.kernel_size)) * mod.in_channels
        add_per_out_pix = 1 if mod.bias is not None else 0
        out_pix = float(numpy.prod(out[0].shape[1:]))

        # total flops counts the cost of summing the multiplications together activation well
        # most implementations and papers do not include this cost
        desc.flops = (mult_per_out_pix + add_per_out_pix) * out_pix
        desc.total_flops = (mult_per_out_pix * 2 + add_per_out_pix) * out_pix

    def _linear_hook(self, mod: Linear, inp: Union[Tuple[Tensor, ...], Tensor], out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'linear', inp, out)

        params = {'weight': mod.weight} if mod.bias is None else {'weight': mod.weight, 'bias': mod.bias}
        prunable_params = {'weight': mod.weight}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        mult_per_out_pix = mod.in_features
        add_per_out_pix = 1 if mod.bias is not None else 0
        out_pix = float(numpy.prod(out[0].shape[1:]))

        # total flops counts the cost of summing the multiplications together activation well
        # most implementations and papers do not include this cost
        desc.flops = (mult_per_out_pix + add_per_out_pix) * out_pix
        desc.total_flops = (mult_per_out_pix * 2 + add_per_out_pix) * out_pix

    def _bn_hook(self, mod: Linear, inp: Union[Tuple[Tensor, ...], Tensor], out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'batch_norm', inp, out)

        params = {'weight': mod.weight} if mod.bias is None else {'weight': mod.weight, 'bias': mod.bias}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        # 4 elementwise operations on the output space, just need to add all of them up
        desc.flops = 4 * float(numpy.prod(out[0].shape[1:]))
        desc.total_flops = desc.flops

    def _pool_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                   out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'pool', inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        flops_per_out_pix = float(numpy.prod(mod.kernel_size) + 1)
        out_pix = float(numpy.prod(out[0].shape[1:]))

        desc.flops = flops_per_out_pix * out_pix
        desc.total_flops = desc.flops

    def _adaptive_pool_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                            out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'global_pool', inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        stride = tuple(inp[0].shape[i] // out[0].shape[i] for i in range(2, len(inp[0].shape)))
        kernel_size = tuple(inp[0].shape[i] - (out[0].shape[i] - 1) * stride[i - 2]
                            for i in range(2, len(inp[0].shape)))
        flops_per_out_pix = float(numpy.prod(kernel_size))
        out_pix = float(numpy.prod(out[0].shape[1:]))

        desc.flops = flops_per_out_pix * out_pix
        desc.total_flops = desc.flops

    def _activation_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                         out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'act', inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        # making assumption that flops spent is one per element (so swish is counted the same activation ReLU)
        desc.flops = float(numpy.prod(out[0].shape[1:]))
        desc.total_flops = desc.flops

    def _softmax_hook(self, mod: Union[_MaxPoolNd, _AvgPoolNd], inp: Union[Tuple[Tensor, ...], Tensor],
                      out: Union[Tuple[Tensor, ...], Tensor]):
        desc, inp, out = self._init_hook(mod, 'softmax', inp, out)

        params = {key: val for key, val in mod.named_parameters()}
        prunable_params = {}

        desc.params = sum([val.numel() for val in params.values()])
        desc.prunable_params = sum([val.numel() for val in prunable_params.values()])
        desc.zeroed_params = sum([(val == 0).sum().item() for val in prunable_params.values()])
        desc.params_dims = {key: tuple(s for s in val.shape) for key, val in params.items()}
        desc.prunable_params_dims = {key: tuple(s for s in val.shape) for key, val in prunable_params.items()}

        flops_per_channel = 2 if len(out[0].shape) < 3 else float(numpy.prod(out[0].shape[2:]))
        desc.flops = flops_per_channel * out[0].shape[1]
        desc.total_flops = desc.flops

    def _init_hook(self, mod: Module, type_: str, inp: Union[Tuple[Tensor, ...], Tensor],
                   out: Union[Tuple[Tensor, ...], Tensor]) -> Tuple[AnalyzedLayerDesc, Tuple[Tensor, ...],
                                                                    Tuple[Tensor, ...]]:
        self._forward_called = True
        self._call_count += 1

        if isinstance(inp, Tensor):
            inp = (inp,)

        if isinstance(out, Tensor):
            out = (out,)

        mod._analyzed_layer_desc = AnalyzedLayerDesc(
            name=mod._analyzed_layer_name, type_=type_, execution_order=self._call_count,
            input_shape=tuple(tuple(ii for ii in i.shape) for i in inp if isinstance(i, Tensor)),
            output_shape=tuple(tuple(oo for oo in o.shape) for o in out if isinstance(o, Tensor))
        )

        return mod._analyzed_layer_desc, inp, out

    @staticmethod
    def _mod_desc(mod: Module) -> AnalyzedLayerDesc:
        children = []
        for _, child in mod.named_modules():
            if child != mod:
                children.append(child)

        merge_descs = [ModuleAnalyzer._mod_desc(child) for child in children]  # type: List[AnalyzedLayerDesc]

        return AnalyzedLayerDesc.merge_descs(mod._analyzed_layer_desc, merge_descs)


def model_ks_desc(analyzer: ModuleAnalyzer) -> List[AnalyzedLayerDesc]:
    descs = []
    layers = {}
    layers.update(get_conv_layers(analyzer.module))
    layers.update(get_linear_layers(analyzer.module))

    for name, _ in layers.items():
        desc = analyzer.layer_desc(name)
        descs.append(desc)

    descs.sort(key=lambda val: val.execution_order)

    return descs


def save_model_ks_desc(analyzer: ModuleAnalyzer, path: str):
    path = clean_path(path)
    create_parent_dirs(path)
    ks_descs = model_ks_desc(analyzer)
    ks_obj = {
        'layer_descriptions': [desc.dict() for desc in ks_descs]
    }

    with open(path, 'w') as file:
        json.dump(ks_obj, file)
