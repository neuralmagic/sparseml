from typing import Union, Tuple
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from ..utils import mask_from_tensor, mask_from_threshold, mask_from_sparsity


__all__ = ['KSLayerMask']


class KSLayerMask(object):
    def __init__(self, layer: Module, param_name: str = 'weight', grad_track_mom: float = -1.0):
        """
        Implementation to apply a kernel sparsity mask to a given param within a layer
        Applies the mask as soon as it is set, as well as anytime apply is called, on the forward, and on the backward

        :param layer: the layer to mask a parameter for
        :param param_name: the name of the parameter within the layer to mask, defaults to weight
        :param grad_track_mom: the momentum to use for tracking the gradients before a mask is applied for the param
                               default of -1.0 will not track the gradients (and anything less than 0)
                               value of 0.0 will only keep the most recent gradient
                               anything up to 1 will follow the momentum formula of G_t = B*G_t-1 + (1-B)*G
        """
        self._layer = layer
        self._param_name = param_name
        self._grad_track_mom = grad_track_mom

        self._param = self._layer.__getattr__(self._param_name)  # type: Parameter
        self._init_param_tensor = self._param.data.clone()  # type: Tensor
        self._unmasked_param_tensor = self._init_param_tensor.clone()  # type: Tensor
        self._grad_set = False
        self._grad = self._init_param_tensor.new_zeros(self._init_param_tensor.shape)  # type: Tensor

        self._enabled = False
        self._mask_tensor = mask_from_tensor(self.param_tensor)
        self._forward_hook = None
        self._gradient_hook = None

    def __del__(self):
        self._delete_hooks()

    @property
    def layer(self):
        return self._layer

    @property
    def param_name(self) -> str:
        return self._param_name

    @property
    def grad_track_mom(self) -> float:
        return self._grad_track_mom

    @grad_track_mom.setter
    def grad_track_mom(self, value: float):
        self._grad_track_mom = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def param(self) -> Parameter:
        return self._param

    @property
    def param_tensor(self) -> Tensor:
        return self._param.data

    @property
    def init_param_tensor(self) -> Tensor:
        return self._init_param_tensor

    @property
    def unmasked_param_tensor(self) -> Tensor:
        return self._unmasked_param_tensor

    @property
    def mask_tensor(self) -> Tensor:
        return self._mask_tensor

    @property
    def grad(self) -> Tensor:
        return self._grad

    @property
    def masked_grad(self) -> Tensor:
        return self._mask_tensor * self._grad

    def enable(self):
        """
        Enable the mask for this layer so that it will begin masking the param and gradients
        """
        self._create_hooks()
        self._enabled = True

    def disable(self):
        """
        Disable the mask for this layer so that it will stop masking the param and gradients
        """
        self._delete_hooks()
        self._enabled = False

        if self._grad_set:
            self._grad = self._grad.new_zeros(self._grad.shape)
            self._grad_set = False

    def apply_mask_tensor_from_threshold(self, threshold: Union[float, Tensor]) -> Tensor:
        """
        Apply a mask to the layer param based on a threshold (anything below threshold is masked, above is not masked)

        :param threshold: the threshold to apply a mask based on, anything less than this value is pruned
        """
        mask_tensor = mask_from_threshold(self.param_tensor, threshold)

        return self.apply_mask_tensor(mask_tensor)

    def apply_mask_tensor_from_sparsity(self, sparsity: float) -> Tensor:
        """
        Apply a mask to the layer param based on a desired amount of sparsity
        Applied by pruning off the smallest N params in the param to reach the given sparsity
        Additionally if the mask has above that sparsity (too man zeros) will randomly select the mask from zeros

        :param sparsity: the desired sparsity to apply a mask based on -- number of desired zeros in the mask
        """
        mask_tensor = mask_from_sparsity(self.param_tensor, sparsity)

        return self.apply_mask_tensor(mask_tensor)

    def apply_mask_tensor(self, mask_tensor: Tensor) -> Tensor:
        """
        Apply a mask to the layer param
        Applies the new mask immediately to the param tensor

        :param mask_tensor: the mask to apply (must be only 1's and 0's)
        :return a tensor of same shape as the param and mask tensor representing the changed parameters
                0 means no change, 1 means masked, -1 means unmasked
        """
        if mask_tensor is None:
            raise ValueError('mask cannot be set to None')

        if mask_tensor.shape != self._param.shape:
            raise ValueError('mask shape of {} does not match layer.param shape of {}'
                             .format(mask_tensor.shape, self._param.shape))

        # figure out the delta of the mask for newly masked and unmasked values
        # we return that tensor so anyone can easily figure out what action happened
        # additionally we need to store the original unmasked values
        newly_masked = ((self._mask_tensor != mask_tensor) & (mask_tensor == 0.0)).type(mask_tensor.type())
        newly_unmasked = ((self._mask_tensor != mask_tensor) & (mask_tensor == 1.0)).type(mask_tensor.type())
        self._unmasked_param_tensor = (newly_masked * self._unmasked_param_tensor +
                                       (newly_masked == 0.0).type(mask_tensor.type()) * self._unmasked_param_tensor)
        self._mask_tensor = mask_tensor
        self._regen_mask()
        self.apply()

        return newly_masked + -1 * newly_unmasked

    def apply_new_tensor(self, param_tensor: Tensor):
        """
        Set the tensor for the parameter
        Will apply the current mask to the param tensor immediately

        :param param_tensor: the tensor to apply to the parameter as a new one
        """
        if param_tensor is None:
            raise ValueError('param_tensor cannot be set to None')

        if param_tensor.shape != self._param.shape:
            raise ValueError('param_tensor shape of {} does not match layer.param shape of {}'
                             .format(param_tensor.shape, self._param.shape))

        self._param.data.copy_(param_tensor)
        self.apply()

    def apply(self):
        """
        Apply the mask to the current params tensor
        Should be called anytime the param changes

        Automatically called before forward (forward_pre_hook)
        gradient is also masked (param hook)
        """
        if not self._enabled:
            return

        if self._param.data.device != self._mask_tensor.device:
            self._regen_mask()

        self._param.data.mul_(self._mask_tensor)

    def _regen_mask(self):
        self._mask_tensor = torch.empty_like(self._param.data).copy_(self._mask_tensor).detach().requires_grad_(False)

    def _create_hooks(self):
        def _mask_forward(_mod: Module, _inp: Union[Tensor, Tuple[Tensor]]):
            self.apply()

        def _mask_gradient(_grad):
            if self._grad_track_mom >= 0.0:
                if not self._grad_set:
                    self._grad.copy_(_grad)
                    self._grad_set = True
                else:
                    self._grad.mul_(self._grad_track_mom).add_((1.0 - self._grad_track_mom) * _grad)

            return _grad.mul(self._mask_tensor)

        if self._forward_hook is None:
            self._forward_hook = self._layer.register_forward_pre_hook(_mask_forward)

        if self._gradient_hook is None:
            self._gradient_hook = self._param.register_hook(_mask_gradient)

    def _delete_hooks(self):
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None

        if self._gradient_hook is not None:
            self._gradient_hook.remove()
            self._gradient_hook = None
