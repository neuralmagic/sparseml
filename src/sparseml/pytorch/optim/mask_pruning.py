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
Code related to applying a mask onto a parameter to impose kernel sparsity,
aka model pruning
"""

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparseml.pytorch.optim.mask_creator_pruning import (
    PruningMaskCreator,
    UnstructuredPruningMaskCreator,
)
from sparseml.pytorch.utils import mask_difference


__all__ = ["ModuleParamPruningMask"]


class ModuleParamPruningMask(object):
    """
    Mask to apply kernel sparsity (model pruning) to a specific parameter in a layer
    """

    def __init__(
        self,
        layer: Module,
        param_name: str = "weight",
        store_init: bool = False,
        store_unmasked: bool = False,
        track_grad_mom: float = -1.0,
        mask_creator: PruningMaskCreator = UnstructuredPruningMaskCreator(),
        layer_name: str = None,
    ):
        """
        :param layer: the layer containing the parameter to mask
        :param param_name: the name of the parameter to mask in the layer,
            default is weight
        :param store_init: store the init weights in a separate variable that can be
            used and referenced later
        :param store_unmasked: store the unmasked weights in a separate variable that
            can be used and referenced later
        :param track_grad_mom: store the gradient updates to the parameter with a
            momentum variable must be in the range [0.0, 1.0), if set to 0.0 then will
            only keep most recent
        :param mask_creator: object to define sparisty mask creation,
            default is unstructured mask
        :param layer_name: the name of the layer the parameter to mask is located in
        """
        self._layer = layer
        self._param_name = param_name
        self._layer_name = layer_name
        self._store_init = store_init
        self._store_unmasked = store_unmasked
        self._track_grad_mom = track_grad_mom
        self._mask_creator = mask_creator

        self._enabled = False
        self._forward_hook = None
        self._gradient_hook = None

        try:
            self._param = self._layer.__getattr__(self._param_name)  # type: Parameter
        except Exception as err:
            raise RuntimeError(
                "Error occurred while trying to get param {} in layer {}: {}".format(
                    self._param_name, self._layer, err
                )
            )

        self._param_mask = torch.ones(self._param.shape)  # initialize to all ones
        self._param_init = None  # type: Tensor
        self._param_unmasked = None  # type: Tensor
        self._param_grad = None  # type: Tensor

        self._setup_param_init()
        self._setup_param_unmasked()
        self._setup_param_grad()

    def __del__(self):
        self._delete_hooks()

    @property
    def layer(self) -> Module:
        """
        :return: the layer containing the parameter to mask
        """
        return self._layer

    @property
    def param_name(self) -> str:
        """
        :return: the name of the parameter to mask in the layer, default is weight
        """
        return self._param_name

    @property
    def layer_name(self) -> str:
        """
        :return: the name of the layer the parameter to mask is located in
        """
        return self._layer_name

    @property
    def name(self) -> str:
        """
        :return: the full name of this sparsity mask in the following format:
            <LAYER>.<PARAM>.sparsity_mask
        """
        return "{}.{}.sparsity_mask".format(self._layer_name, self._param_name)

    @property
    def store_init(self) -> bool:
        """
        :return: store the init weights in a separate variable that can be used and
            referenced later
        """
        return self._store_init

    @property
    def store_unmasked(self) -> bool:
        """
        :return: store the unmasked weights in a separate variable that can be used and
            referenced later
        """
        return self._store_unmasked

    @property
    def track_grad_mom(self) -> float:
        """
        :return: store the gradient updates to the parameter with a momentum variable
            must be in the range [0.0, 1.0), if set to 0.0 then will only
            keep most recent
        """
        return self._track_grad_mom

    @property
    def mask_creator(self) -> PruningMaskCreator:
        """
        :return: SparsityMaskCreator object used to generate masks
        """
        return self._mask_creator

    @property
    def enabled(self) -> bool:
        """
        :return: True if the parameter is currently being masked, False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to mask the parameter, False otherwise
        """
        if value and not self._enabled:
            self._create_hooks()
            self._param_grad = None
            self._setup_param_grad()
        elif not value and self._enabled:
            self._delete_hooks()

        self._enabled = value

    @property
    def param_data(self) -> Tensor:
        """
        :return: the current tensor in the parameter
        """
        return self._param.data

    @property
    def param_mask(self) -> Tensor:
        """
        :return: the current mask applied to the parameter
        """
        return self._param_mask

    @property
    def param_init(self) -> Union[None, Tensor]:
        """
        :return: the initial value of the parameter before being masked
        """
        return self._param_init

    @property
    def param_unmasked(self) -> Union[None, Tensor]:
        """
        :return: the unmasked value of the parameter
            (stores the last unmasked value before masking)
        """
        if self._param_unmasked is None:
            return None

        return (
            self._param.data
            + (self._param_mask == 0.0).type(self._param.data.type())
            * self._param_unmasked
        )

    @property
    def param_grad(self) -> Union[None, Tensor]:
        """
        :return: the current gradient values for the parameter
        """
        return self._param_grad

    def set_param_data(self, value: Tensor):
        """
        :param value: the value to set as the current tensor for the parameter,
            if enabled the mask will be applied
        """
        if value is None:
            raise ValueError("param_data cannot be set to None")

        if value.shape != self._param.data.shape:
            raise ValueError(
                f"param_tensor shape of {value.shape} does not match layer.param "
                f"shape of {self._param.shape}"
            )

        value = self._check_regen_value(value)
        self._check_regen_param_vals()
        self._param.data.copy_(value)
        self._param_unmasked = None
        self._setup_param_unmasked()
        self.apply()

    def set_param_mask(self, value: Tensor):
        """
        :param value: the mask to set and apply as the current tensor,
            if enabled mask is applied immediately
        """
        if value is None:
            raise ValueError("mask cannot be set to None")

        if value.shape != self._param.shape:
            raise ValueError(
                "mask shape of {} does not match layer.param shape of {}".format(
                    value.shape, self._param.shape
                )
            )

        value = self._check_regen_value(value)
        self._check_regen_param_vals()
        mask_diff = mask_difference(self._param_mask, value)

        if self._param_unmasked is not None:
            # store our unmasked values if they should be tracked
            # we only want to update our param_masked tensor with the ones
            # that are newly masked
            self._param_unmasked = (mask_diff == -1.0).type(
                self._param.data.type()
            ) * self._param.data + (mask_diff != -1.0).type(
                self._param.data.type()
            ) * self._param_unmasked
        self._param_mask = value
        self.apply()

        return mask_diff

    def set_param_mask_from_weights(self) -> Tensor:
        """
        Convenience function to set the parameter mask such that the
        mask is 1 if the parameter value is non zero and 0 otherwise,
        unless otherwise defined by this object's mask_creator.

        """
        value = self._mask_creator.create_sparsity_mask_from_tensor(self._param.data)

        return self.set_param_mask(value)

    def set_param_mask_from_abs_threshold(
        self, threshold: Union[float, Tensor]
    ) -> Tensor:
        """
        Convenience function to set the parameter mask such that if
        abs(value) <= threshold the it is masked to 0

        :param threshold: the threshold at which all values will be masked to 0
        """
        value = self._mask_creator.create_sparsity_mask_from_abs_threshold(
            self._param.data, threshold
        )

        return self.set_param_mask(value)

    def set_param_mask_from_sparsity(self, sparsity: float) -> Tensor:
        """
        Convenience function to set the parameter mask such that it has a specific
        amount of masked values such that the percentage equals the sparsity amount
        given. Masks the absolute smallest values up until sparsity is reached.

        :param sparsity: the decimal sparsity to set the param mask to
        """
        value = self._mask_creator.create_sparsity_mask(self._param.data, sparsity)

        return self.set_param_mask(value)

    def apply(self):
        """
        apply the current mask to the params tensor (zero out the desired values)
        """
        if not self._enabled:
            return

        self._check_regen_param_vals()

        with torch.no_grad():
            self._param.data.mul_(self._param_mask)

    def reset(self):
        """
        resets the current stored tensors such that they will be on the same device
        and have the proper data
        """
        self._check_regen_param_vals()
        self._param.data.copy_(self._param_init)

    def _check_regen_value(self, val: Tensor) -> Tensor:
        if self._param.data.device != val.device:
            val = ModuleParamPruningMask._detach_tens(
                torch.empty_like(self._param.data).copy_(val)
            )

        return val

    def _check_regen_param_vals(self):
        if self._param.data.device != self._param_mask.device:
            self._param_mask = ModuleParamPruningMask._detach_tens(
                torch.empty_like(self._param.data).copy_(self._param_mask)
            )

        if (
            self._param_init is not None
            and self._param.data.device != self._param_init.device
        ):
            self._param_init = ModuleParamPruningMask._detach_tens(
                torch.empty_like(self._param.data).copy_(self._param_init)
            )

        if (
            self._param_unmasked is not None
            and self._param.data.device != self._param_unmasked.device
        ):
            self._param_unmasked = ModuleParamPruningMask._detach_tens(
                torch.empty_like(self._param.data).copy_(self._param_unmasked)
            )

        if (
            self._param_grad is not None
            and self._param.data.device != self._param_grad.device
        ):
            self._param_grad = ModuleParamPruningMask._detach_tens(
                torch.empty_like(self._param.data).copy_(self._param_grad)
            )

    def _create_hooks(self):
        if self._forward_hook is None:
            self._forward_hook = self._layer.register_forward_pre_hook(
                self._hook_mask_forward
            )

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
        if 0.0 <= self._track_grad_mom < 1.0:
            self._param_grad.mul_(self._track_grad_mom).add_(
                (1.0 - self._track_grad_mom) * grad
            )

        return grad.mul_(self._param_mask)

    def _setup_param_init(self):
        if self._store_init and self._param_init is None:
            self._param_init = ModuleParamPruningMask._detach_tens(
                self._param.data.clone()
            )
        elif not self._store_init and self._param_init is not None:
            self._param_init = None

    def _setup_param_unmasked(self):
        if self._store_unmasked and self._param_unmasked is None:
            self._param_unmasked = ModuleParamPruningMask._detach_tens(
                self._param.data.clone()
            )
        elif not self._store_unmasked and self._param_unmasked is not None:
            self._param_unmasked = None

    def _setup_param_grad(self):
        if self._track_grad_mom >= 0.0 and self._param_grad is None:
            self._param_grad = ModuleParamPruningMask._detach_tens(
                self._param.data.new_zeros(self._param.data.shape)
            )
        elif self._track_grad_mom < 0.0 and self._param_grad is not None:
            self._param_grad = None

    @staticmethod
    def _detach_tens(tens) -> Tensor:
        return tens.detach().requires_grad_(False)
