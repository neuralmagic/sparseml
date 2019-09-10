from typing import Union, Callable, Any, Tuple
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


__all__ = ['ASLayerTracker']


class ASLayerTracker(object):
    def __init__(self, layer: Module, track_input: bool = False, track_output: bool = False,
                 input_func: Union[None, Callable] = None, output_func: Union[None, Callable] = None):
        super().__init__()
        self._layer = layer
        self._track_input = track_input
        self._track_output = track_output
        self._input_func = input_func
        self._output_func = output_func

        self._enabled = False
        self._tracked_input = {}
        self._tracked_output = {}
        self._hook_handle = None  # type: RemovableHandle

    def __del__(self):
        self._disable_hooks()

    def enable(self):
        if not self._enabled:
            self._enabled = True
            self._enable_hooks()

        self.clear()

    def disable(self):
        if self._enabled:
            self._enabled = False
            self._disable_hooks()

        self.clear()

    def clear(self):
        self._tracked_input.clear()
        self._tracked_output.clear()

    @property
    def tracked_input(self):
        return self._tracked_input

    @property
    def tracked_output(self):
        return self._tracked_output

    def _enable_hooks(self):
        if self._hook_handle is not None:
            return

        def _forward_hook(_mod: Module, _inp: Union[Tensor, Tuple[Tensor]], _out: Union[Tensor, Tuple[Tensor]]):
            if self._track_input:
                tracked = _inp

                if self._input_func is not None:
                    tracked = self._input_func(_inp)

                key = 'cpu' if not tracked.is_cuda else 'cuda:{}'.format(tracked.get_device())
                self._tracked_input[key] = tracked

            if self._track_output:
                tracked = _out

                if self._output_func is not None:
                    tracked = self._output_func(_out)

                key = 'cpu' if not tracked.is_cuda else 'cuda:{}'.format(tracked.get_device())
                self._tracked_output[key] = tracked

        self._hook_handle = self._layer.register_forward_hook(_forward_hook)

    def _disable_hooks(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
