from typing import Union, Tuple
import numpy
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


__all__ = ['FlopsAnalyzerModule']


class FlopsAnalyzerModule(Module):
    def __init__(self, module: Module, actual: bool = True):
        """
        Analyzer to calculate the number of FLOPS and params in a module
        An inference run of the network must happen before the numbers are available
        Works by hooking on to the forward calls of each module that can be calculated for and then storing the results

        :param module: The module to track flops for
        :param actual: True if tracking actual calculated flops, False if tracking according to what most papers do
                       (generally papers and other imp skip the cost of summing the result after mult weights)
        """
        super(FlopsAnalyzerModule, self).__init__()
        self.module = module
        self._actual = actual
        self._hooks = {}
        self._forward_called = False

        def _module_forward(_mod, _inp, _out):
            self._forward_called = True

        self._module_hook = self.module.register_forward_hook(_module_forward)

        for name, mod in self.module.named_modules():
            hook = self._add_hook(name, mod)

            if hook is not None:
                self._hooks[name] = hook

    def __del__(self):
        self.disable()

    @property
    def total_flops(self) -> int:
        """
        :return: total flops calculated for the module for the given batches called into the model
        """
        return self.layer_flops(None)

    @property
    def total_params(self) -> int:
        """
        :return: total number of params in the module
        """
        return self.layer_params(None)

    def disable(self):
        for hook in self._hooks.values():
            hook.remove()

        self._hooks.clear()

        if self._module_hook is not None:
            self._module_hook.remove()
            self._module_hook = None

    def layer_flops(self, name: Union[str, None]) -> int:
        """
        :param name: name of the layer to get the flops for, None for total of module
        :return: the calculated FLOPS for the given layer
        """
        if not self._forward_called:
            raise RuntimeError('Must first execute model for an inference batch to get flops count')

        mod = self.module

        if name:
            for lay in name.split('.'):
                mod = mod.__getattr__(lay)

        flops = 0

        for _, child in mod.named_modules():
            flops += child.analyzed_flops

        return flops

    def layer_params(self, name: Union[str, None]) -> int:
        """
        :param name: name of the layer to get the flops for, None for total of module
        :return: the total number of params for the given layer
        """
        if not self._forward_called:
            raise RuntimeError('Must first execute model for an inference batch to get params count')

        mod = self.module

        if name:
            for lay in name.split('.'):
                mod = mod.__getattr__(lay)

        params = 0

        for _, child in mod.named_modules():
            params += child.analyzed_params

        return params

    def forward(self, *inp):
        self.module(*inp)

    def _add_hook(self, name, mod: Module) -> Union[None, RemovableHandle]:
        mod.analyzed_name = name
        mod.analyzed_flops = 0
        mod.analyzed_params = 0

        if isinstance(mod, _ConvNd):
            return self._add_conv_hook(mod)

        if isinstance(mod, Linear):
            return self._add_linear_hook(mod)

        if isinstance(mod, _BatchNorm):
            return self._add_bn_hook(mod)

        if isinstance(mod, _MaxPoolNd) or isinstance(mod, _AvgPoolNd):
            return self._add_pool_hook(mod)

        if isinstance(mod, _AdaptiveAvgPoolNd) or isinstance(mod, _AdaptiveMaxPoolNd):
            return self._add_adaptive_pool_hook(mod)

        if (isinstance(mod, Threshold) or isinstance(mod, ReLU) or isinstance(mod, ReLU6) or isinstance(mod, RReLU) or
                isinstance(mod, LeakyReLU) or isinstance(mod, PReLU) or isinstance(mod, ELU) or isinstance(mod, CELU) or
                isinstance(mod, SELU) or isinstance(mod, GLU) or isinstance(mod, Hardtanh) or isinstance(mod, Tanh) or
                isinstance(mod, Sigmoid) or isinstance(mod, LogSigmoid)):
            return self._add_activation_hook(mod)

        if isinstance(mod, Softmax) or isinstance(mod, Softmax2d):
            return self._add_softmax_hook(mod)

        return None

    def _add_conv_hook(self, conv: _ConvNd) -> RemovableHandle:
        def _hook(_mod: _ConvNd, _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            batch_size = _out.shape[0]
            flops_per_out_pixel = numpy.prod(_mod.kernel_size) * _mod.in_channels

            if self._actual:
                # count the cost of summing the multiplications together as well
                # most implementations and papers do not include this cost
                flops_per_out_pixel *= 2

            if _mod.bias is not None:
                flops_per_out_pixel += 1

            out_pixels_per_batch = numpy.prod(_out.shape[1:])
            flops_per_batch = flops_per_out_pixel * out_pixels_per_batch

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = _mod.weight.numel()

            if _mod.bias is not None:
                _mod.analyzed_params += _mod.bias.numel()

        return conv.register_forward_hook(_hook)

    def _add_linear_hook(self, linear: Linear) -> RemovableHandle:
        def _hook(_mod: Linear, _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            batch_size = _out.shape[0]
            flops_per_out_pixel = _mod.in_features

            if self._actual:
                # count the cost of summing the multiplications together as well
                # most implementations and papers do not include this cost
                flops_per_out_pixel *= 2

            if _mod.bias is not None:
                flops_per_out_pixel += 1

            out_pixels_per_batch = _mod.out_features
            flops_per_batch = flops_per_out_pixel * out_pixels_per_batch

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = _mod.weight.numel()

            if _mod.bias is not None:
                _mod.analyzed_params += _mod.bias.numel()

        return linear.register_forward_hook(_hook)

    def _add_bn_hook(self, batch_norm: _BatchNorm) -> RemovableHandle:
        def _hook(_mod: _BatchNorm, _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            batch_size = _out.shape[0]
            # 4 elementwise operations on the output space, just need to add all of them up
            flops_per_batch = 4 * numpy.prod(_out.shape[1:])

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = _mod.weight.numel() + _mod.bias.numel()

        return batch_norm.register_forward_hook(_hook)

    def _add_pool_hook(self, pool: Union[_MaxPoolNd, _AvgPoolNd]) -> RemovableHandle:
        def _hook(_mod: Union[_MaxPoolNd, _AvgPoolNd],
                  _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            batch_size = _out.shape[0]
            flops_per_out_pixel = (numpy.prod(_mod.kernel_size) * _out.shape[1]) + 1
            out_pixels_per_batch = numpy.prod(_out.shape[1:])
            flops_per_batch = flops_per_out_pixel * out_pixels_per_batch

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = sum(param.numel() for param in _mod.parameters())

        return pool.register_forward_hook(_hook)

    def _add_adaptive_pool_hook(self, pool: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd]) -> RemovableHandle:
        def _hook(_mod: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd],
                  _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            if not isinstance(_inp, Tensor):
                _inp = _inp[0]

            batch_size = _out.shape[0]

            # make an assumption that we are averaging down to 1 on the output size
            flops_per_batch = numpy.prod(_inp.shape[1:])

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = sum(param.numel() for param in _mod.parameters())

        return pool.register_forward_hook(_hook)

    def _add_activation_hook(self, act: Module) -> RemovableHandle:
        def _hook(_mod: Module, _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            batch_size = _out.shape[0]

            # making assumption that flops spent is one per element (so swish is counted the same as ReLU)
            flops_per_batch = numpy.prod(_out.shape[1:])

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = sum(param.numel() for param in _mod.parameters())

        return act.register_forward_hook(_hook)

    def _add_softmax_hook(self, soft: Union[Softmax, Softmax2d]):
        def _hook(_mod: Module, _inp: Union[Tuple[Tensor, ...], Tensor], _out: Union[Tuple[Tensor, ...], Tensor]):
            batch_size = _out.shape[0]

            # making assumption that flops spent is one per element (so swish is counted the same as ReLU)
            flops_per_channel = 2 if len(_out.shape) < 3 else numpy.prod(_out.shape[2:])
            flops_per_batch = flops_per_channel * _out.shape[1]

            _mod.analyzed_flops = flops_per_batch * batch_size
            _mod.analyzed_params = sum(param.numel() for param in _mod.parameters())

        return soft.register_forward_hook(_hook)
