from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from ..utils import mask_from_tensor, mask_from_threshold, mask_from_sparsity


__all__ = ['KSLayerParamMask']


class KSLayerParamMask(object):
    def __init__(self, layer: Module, param_name: str = 'weight',
                 store_init: bool = True, store_unmasked: bool = True, track_grad_mom: float = 0.9):
        self._layer = layer
        self._param_name = param_name
        self._store_init = store_init
        self._store_unmasked = store_unmasked
        self._track_grad_mom = track_grad_mom

        self._enabled = False
        self._forward_hook = False
        self._gradient_hook = False

        try:
            self._param = self._layer.__getattr__(self._param_name)  # type: Parameter
        except Exception as err:
            raise RuntimeError('Error occurred while trying to get param {} in layer {}: {}'
                               .format(self._param_name, self._layer, err))

        self._param_mask = mask_from_tensor(self._param)  # type: Tensor
        self._param_init = None  # type: Tensor
        self._param_unmasked = None  # type: Tensor
        self._param_grad = None  # type: Tensor
        
        self._setup_param_grad()
        self._setup_param_unmasked()
        self._setup_param_grad()
        
    @property
    def layer(self) -> Module:
        return self._layer
    
    @property
    def param_name(self) -> str:
        return self._param_name
    
    @property
    def store_init(self) -> bool:
        return self._store_init
    
    @store_init.setter
    def store_init(self, value: bool):
        self._store_init = value
        self._setup_param_init()

    @property
    def store_unmasked(self) -> bool:
        return self._store_unmasked

    @store_unmasked.setter
    def store_unmasked(self, value: bool):
        self._store_unmasked = value
        self._setup_param_unmasked()

    @property
    def track_grad_mom(self) -> float:
        return self._track_grad_mom

    @track_grad_mom.setter
    def track_grad_mom(self, value: float):
        self._track_grad_mom = value
        self._setup_param_grad()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value and not self._enabled:
            self._create_hooks()
            self._param_grad = None
            self._setup_param_grad()
        elif not value and self._enabled:
            self._delete_hooks()

        self._enabled = value
        
    @property
    def param_data(self) -> Tensor:
        return self._param.data

    @property
    def param_mask(self) -> Tensor:
        return self._param_mask

    @property
    def param_init(self) -> Union[None, Tensor]:
        return self._param_init

    @property
    def param_unmasked(self) -> Union[None, Tensor]:
        if self._param_unmasked is None:
            return None

        return self._param.data + (self._param_mask == 0.0).type(self._param.data.type()) * self._param_unmasked

    @property
    def param_grad(self) -> Union[None, Tensor]:
        return self._param_grad

    def set_param_data(self, value: Tensor):
        if value is None:
            raise ValueError('param_data cannot be set to None')

        if value.shape != self._param.data.shape:
            raise ValueError('param_tensor shape of {} does not match layer.param shape of {}'
                             .format(value.shape, self._param.shape))

        self._param.data.copy_(value)
        self._param_unmasked = None
        self._setup_param_unmasked()
        self.apply()

    def set_param_mask(self, value: Tensor):
        if value is None:
            raise ValueError('mask cannot be set to None')

        if value.shape != self._param.shape:
            raise ValueError('mask shape of {} does not match layer.param shape of {}'
                             .format(value.shape, self._param.shape))

        # figure out the delta of the mask for newly masked and unmasked values for returning the info
        # also add the values that are newly masked to our unmasked tensor
        newly_masked = ((self._param_mask != value) & (value == 0.0)).type(self._param.data.type())
        newly_unmasked = ((self._param_mask != value) & (value == 1.0)).type(self._param.data.type())
        self._param_unmasked = (newly_masked * self._param_unmasked +
                                (newly_masked == 0.0).type(self._param.data.type()) * self._param_unmasked)
        self._param_mask = value
        self.apply()

        return newly_masked + -1 * newly_unmasked

    def set_param_mask_from_threshold(self, threshold: Union[float, Tensor]) -> Tensor:
        value = mask_from_threshold(self._param.data, threshold)

        return self.set_param_mask(value)

    def set_param_mask_from_sparsity(self, sparsity: float) -> Tensor:
        value = mask_from_sparsity(self._param.data, sparsity)

        return self.set_param_mask(value)

    def cycle_param_mask(self, cycle_percent: float) -> Tensor:
        raise NotImplementedError()
    
    def apply(self):
        if not self._enabled:
            return

        if self._param.data.device != self._param_mask.device:
            # param is on a different device, regen values so all tensors are on the same device
            self._regen_param_vals()

        with torch.no_grad():
            self._param.data.mul_(self._param_mask)

    def _regen_param_vals(self):
        self._param_mask = KSLayerParamMask._detach_tens(torch.empty_like(self._param.data).copy_(self._param_mask))

        if self._param_init is not None:
            self._param_init = KSLayerParamMask._detach_tens(torch.empty_like(self._param.data).copy_(self._param_init))

        if self._param_unmasked is not None:
            self._param_unmasked = KSLayerParamMask._detach_tens(torch.empty_like(self._param.data)
                                                                 .copy_(self._param_unmasked))

        if self._param_grad is not None:
            self._param_grad = KSLayerParamMask._detach_tens(torch.empty_like(self._param.data).copy_(self._param_grad))

    def _create_hooks(self):
        if self._forward_hook is None:
            self._forward_hook = self._layer.register_forward_pre_hook(self._hook_mask_forward)

        if self._gradient_hook is None:
            self._gradient_hook = self._param.register_hook(self._hook_mask_gradient)

    def _delete_hooks(self):
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None

        if self._gradient_hook is not None:
            self._gradient_hook.remove()
            self._gradient_hook = None

    def _hook_mask_forward(self, mod: Module, inp: Union[Tensor, Tuple[Tensor]]):
        self.apply()

    def _hook_mask_gradient(self, grad):
        if self._track_grad_mom >= 0.0:
            self._param_grad.mul_(self._track_grad_mom).add_((1.0 - self._track_grad_mom) * grad)

        return grad.mul_(self._param_mask)
        
    def _setup_param_init(self):
        if self._store_init and self._param_init is None:
            self._param_init = KSLayerParamMask._detach_tens(self._param.data.clone())
        elif not self._store_init and self._param_init is not None:
            self._param_init = None
            
    def _setup_param_unmasked(self):
        if self._store_unmasked and self._param_unmasked is None:
            self._param_unmasked = KSLayerParamMask._detach_tens(self._param.data.clone())
        elif not self._store_unmasked and self._param_unmasked is not None:
            self._param_unmasked = None
        
    def _setup_param_grad(self):
        if self._track_grad_mom >= 0.0 and self._param_grad is None:
            self._param_grad = KSLayerParamMask._detach_tens(self._param.data.new_zeros(self._param.data.shape))
        elif self._track_grad_mom < 0.0 and self._param_grad is not None:
            self._param_grad = None

    @staticmethod
    def _detach_tens(tens) -> Tensor:
        return tens.detach().requires_grad_(False)
