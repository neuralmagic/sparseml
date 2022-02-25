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

NOTE: this file is in the process of being phased out in favor of the
sparsification package. Once all references to mask utils in the optim
package are migrated, this file will be deleted
"""

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparseml.pytorch.optim.mask_creator_pruning import (
    PruningMaskCreator,
    UnstructuredPruningMaskCreator,
)
from sparseml.pytorch.optim.mask_pruning_scorer import create_pruning_param_scorer
from sparseml.pytorch.utils import mask_difference


__all__ = [
    "ModuleParamPruningMask",
]


class ModuleParamPruningMask(object):
    """
    Mask to apply kernel sparsity (model pruning) to a specific parameter in a layer

    :param layers: the layers containing the parameters to mask
    :param param_names: the names of the parameter to mask in each layer. If only
        one name is given, that name will be applied to all layers that this object
        masks. default is weight
    :param store_init: store the init weights in a separate variable that can be
        used and referenced later
    :param store_unmasked: store the unmasked weights in a separate variable that
        can be used and referenced later
    :param track_grad_mom: store the gradient updates to the parameter with a
        momentum variable must be in the range [0.0, 1.0), if set to 0.0 then will
        only keep most recent
    :param mask_creator: object to define sparisty mask creation,
        default is unstructured mask
    :param layer_names: the name of the layers the parameters to mask are located in
    :param global_sparsity: set True to enable global pruning. if True, when creating
        sparsity masks for a target sparsity sparsity masks will be created such that
        the average sparsity across all given layers is the target sparsity with the
        lowest global values masked. If False, each layer will be masked to the target
        sparsity ranking values within each individual tensor. Default is False
    :param score_type: the method used to score parameters for masking, i.e.
        'magnitude', 'movement'. Default is 'magnitude'
    """

    def __init__(
        self,
        layers: List[Module],
        param_names: Union[str, List[str]] = "weight",
        store_init: bool = False,
        store_unmasked: bool = False,
        track_grad_mom: float = -1.0,
        mask_creator: PruningMaskCreator = UnstructuredPruningMaskCreator(),
        layer_names: Optional[List[str]] = None,
        global_sparsity: bool = False,
        score_type: str = "magnitude",
    ):
        self._layers = layers
        self._param_names = (
            param_names
            if isinstance(param_names, List)
            else [param_names] * len(self._layers)
        )
        self._layer_names = layer_names
        self._store_init = store_init
        self._store_unmasked = store_unmasked
        self._track_grad_mom = track_grad_mom
        self._mask_creator = mask_creator
        self._global_sparsity = global_sparsity

        self._enabled = False
        self._forward_hooks = [None] * len(self._layers)
        self._undo_mask_hooks = [None] * len(self._layers)
        self._gradient_hooks = [None] * len(self._layers)

        self._params = []  # type: List[Parameter]
        self._param_orig_sizes = []  # type: List[int]
        for layer, param_name in zip(self._layers, self._param_names):
            try:
                layer_param = layer.__getattr__(param_name)
                self._params.append(layer_param)
                self._param_orig_sizes.append(layer_param.numel())
            except Exception as err:
                raise RuntimeError(
                    f"Error occurred while trying to get param {param_name} "
                    f"in layer {layer}: {err}"
                )

        # initialize masks to all ones
        self._param_masks = [torch.ones(param.shape) for param in self._params]
        self._params_init = [None] * len(self._layers)  # type: List[Tensor]
        self._params_unmasked = [None] * len(self._layers)  # type: List[Tensor]
        self._params_grad = [None] * len(self._layers)  # type: List[Tensor]
        self._params_movement = [None] * len(self._layers)  # type: List[Tensor]
        self._params_applied_thinning = [0.0] * len(self._layers)  # type: List[float]

        # create scorer
        self._scorer = create_pruning_param_scorer(self._params, score_type)

        # movement pruning requires weight reintroduction
        self._allow_reintroduction = self._scorer.get_name() == "movement"
        self._store_unmasked |= self._allow_reintroduction  # to support reintroduction

        self._setup_params_init()
        self._setup_params_unmasked()
        self._setup_params_grad()

    def __len__(self):
        return len(self._layers)

    def __del__(self):
        self._delete_hooks()

    @property
    def layers(self) -> List[Module]:
        """
        :return: the layers containing the parameters to mask
        """
        return self._layers

    @property
    def param_names(self) -> List[str]:
        """
        :return: the names of the parameters to mask in the layers
        """
        return self._param_names

    @property
    def layer_names(self) -> Optional[List[str]]:
        """
        :return: the names of the layers the parameter to mask is located in
        """
        return self._layer_names

    @property
    def names(self) -> List[str]:
        """
        :return: the full names of the sparsity masks in the following format:
            <LAYER>.<PARAM>.sparsity_mask
        """
        return [
            f"{layer_name}.{param_name}.sparsity_mask"
            for layer_name, param_name in zip(self._layer_names, self._param_names)
        ]

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
    def global_sparsity(self) -> bool:
        """
        :return: True if global pruning is enabled, False otherwise
        """
        return self._global_sparsity

    @property
    def enabled(self) -> bool:
        """
        :return: True if the parameter is currently being masked, False otherwise
        """
        return self._enabled

    @property
    def allow_reintroduction(self) -> bool:
        """
        :return: True if weight reintroduction is allowed
        """
        return self._allow_reintroduction

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to mask the parameter, False otherwise
        """
        if value and not self._enabled:
            self._create_hooks()
            self._params_grad = [None] * len(self._params)
            self._setup_params_grad()
        elif not value and self._enabled:
            self._delete_hooks()

        self._enabled = value

    @property
    def params_data(self) -> List[Tensor]:
        """
        :return: the current tensors in each of the parameters
        """
        return [param.data for param in self._params]

    @property
    def param_masks(self) -> List[Tensor]:
        """
        :return: the current masks applied to each of the parameters
        """
        return self._param_masks

    @property
    def params_init(self) -> List[Optional[Tensor]]:
        """
        :return: the initial values of the parameters before being masked
        """
        return self._params_init

    @property
    def params_unmasked(self) -> List[Optional[Tensor]]:
        """
        :return: the unmasked values of the parameters
            (stores the last unmasked value before masking)
        """
        params_unmasked = []
        for idx in range(len(self._params)):
            if self._params_unmasked[idx] is None:
                params_unmasked.append(None)
            else:
                params_unmasked.append(
                    self._params[idx].data
                    + (self._param_masks == 0.0).type(self._params[idx].data.type())
                    * self._params_unmasked[idx]
                )
        return params_unmasked

    @property
    def params_grad(self) -> List[Optional[Tensor]]:
        """
        :return: the current gradient values for each parameter
        """
        return self._params_grad

    @property
    def score_type(self) -> str:
        """
        :return: the scoring method used to create masks (i.e. magnitude, movement)
        """
        return self._scorer.get_name()

    def set_param_data(self, value: Tensor, param_idx: int):
        """
        :param value: the value to set as the current tensor for the parameter,
            if enabled the mask will be applied
        :param param_idx: index of the parameter in this object to set the data of
        """
        if value is None:
            raise ValueError("param data cannot be set to None")

        if value.shape != self._params[param_idx].data.shape:
            raise ValueError(
                f"param_tensor shape of {value.shape} does not match parameter "
                f"shape of {self._param[param_idx].shape}"
            )

        value = self._check_regen_value(value, param_idx)
        self._check_regen_param_vals(param_idx)
        self._params[param_idx].data.copy_(value)
        self._params_unmasked[param_idx] = None
        self._setup_params_unmasked(param_idx)

        if not self._allow_reintroduction:
            self.apply(param_idx)

    def set_param_masks(self, masks: List[Tensor]):
        """
        :param masks: the masks to set and apply as the current param tensors,
            if enabled mask is applied immediately
        """
        mask_diffs = []
        for idx, value in enumerate(masks):
            if value is None:
                raise ValueError("mask cannot be set to None")

            if value.shape != self._params[idx].shape:
                raise ValueError(
                    "mask shape of {} does not match layer.param shape of {}".format(
                        value.shape, self._params[idx].shape
                    )
                )

            value = self._check_regen_value(value, idx)
            self._check_regen_param_vals(idx)
            mask_diff = mask_difference(self._param_masks[idx], value)

            self._param_masks[idx] = value

            mask_diffs.append(mask_diff)

        self._scorer.mask_update(masks, mask_diffs)

        if not self._allow_reintroduction:
            self.apply()

        return mask_diffs

    def set_param_masks_from_weights(self) -> List[Tensor]:
        """
        Convenience function to set the parameter masks such that the
        mask is 1 if a parameter value is non zero and 0 otherwise,
        unless otherwise defined by this object's mask_creator
        """
        masks = self._mask_creator.create_sparsity_masks_from_tensor(
            [param.data for param in self._params]
        )

        return self.set_param_masks(masks)

    def set_param_masks_from_abs_threshold(
        self, threshold: Union[float, Tensor]
    ) -> List[Tensor]:
        """
        Convenience function to set the parameter masks such that if
        abs(value) <= threshold the it a value is masked to 0

        :param threshold: the threshold at which all values will be masked to 0
        """
        param_scores = self._scorer.score_parameters()
        masks = self._mask_creator.create_sparsity_masks_from_threshold(
            param_scores, threshold
        )

        return self.set_param_masks(masks)

    def set_param_masks_from_sparsity(
        self, sparsity: Union[float, List[float]]
    ) -> List[Tensor]:
        """
        Convenience function to set the parameter masks such that each masks have an
        amount of masked values such that the percentage equals the sparsity amount
        given. Masks the absolute smallest values up until sparsity is reached.

        :param sparsity: the decimal sparsity to set the param mask to can also be a
            list where each element is a sparsity for a tensor in the same position in
            the tensor list. If global sparsity is enabled, all values of the sparsity
            list must be the same
        """
        param_scores = self._scorer.score_parameters()

        if isinstance(sparsity, float):
            sparsity = [sparsity] * len(self._params)
        for idx, sparsity_val in enumerate(sparsity):
            applied_thinning = self._params_applied_thinning[idx]
            if applied_thinning > 0.0:
                # adjust sparsity for thinned (compressed) layer param
                # derived from:
                # remaining_num_els * (1 - adjusted_sparsity) = \
                #   orig_num_els * (1 - sparsity)
                # with applied_thinning = 1 - (remaining_num_els / orig_num_els)
                sparsity[idx] = (sparsity_val - applied_thinning) / (
                    1 - applied_thinning
                )

        masks = self._mask_creator.create_sparsity_masks(
            param_scores, sparsity, global_sparsity=self._global_sparsity
        )
        self._scorer.update_last_applied_sparsity(sparsity)

        return self.set_param_masks(masks)

    def apply(self, param_idx: Optional[int] = None):
        """
        apply the current mask to the params tensor (zero out the desired values)

        :param param_idx: index of parameter to apply mask to. if not set, then masks
            will be applied to all parameters with available masks
        """
        if not self._enabled:
            return

        indices = range(len(self._params)) if param_idx is None else [param_idx]

        for idx in indices:
            self._check_regen_param_vals(idx)

            with torch.no_grad():
                if self._store_unmasked:
                    self._params_unmasked[idx] = self._params[idx].data.mul(
                        1 - self._param_masks[idx]  # inverted mask
                    )
                self._params[idx].data.mul_(self._param_masks[idx])

    def reset(self):
        """
        resets the current stored tensors such that they will be on the same device
        and have the initial data
        """
        self._check_regen_param_vals()
        for idx, param in enumerate(self._params):
            param.data.copy_(self._params_init[idx])

    def pre_optim_step_update(self):
        """
        updates scores and buffers that depend on gradients. Should be called
        before Optimizer.step() to grab the latest gradients
        """
        self._scorer.pre_optim_step_update(self._param_masks)

    def pruning_end(self, leave_enabled: bool):
        """
        Performs any cleanup necessary for this pruning method.
        Disables weight reintroduction if enabled and applies masks

        :param leave_enabled: if False, all pruning hooks will be destroyed. Default
            is True
        """
        if not leave_enabled:
            self.enabled = False
        self._allow_reintroduction = False
        self.apply()  # ensure that weights are pruned to final level
        self._scorer.on_pruning_end()

    def disable_reintroduction(self):
        """
        if weight reintroduction is enabled (only during movement pruning),
        disables further weight reintroduction
        """
        self._allow_reintroduction = False

    def _check_regen_value(self, val: Tensor, param_idx: int) -> Tensor:
        if self._params[param_idx].data.device != val.device:
            val = ModuleParamPruningMask._detach_tens(
                torch.empty_like(self._params[param_idx].data).copy_(val)
            )

        return val

    def _check_regen_param_vals(self, param_idx: int = None):
        indices = range(len(self._params)) if param_idx is None else [param_idx]

        for idx in indices:
            if self._params[idx].data.numel() < self._param_masks[idx].numel():
                self._params_applied_thinning[idx] = 1 - (
                    self._params[idx].data.numel() / self._param_orig_sizes[idx]
                )
                self._param_masks[idx] = ModuleParamPruningMask._detach_tens(
                    torch.ones_like(self._params[idx].data)
                )
            if self._params[idx].data.device != self._param_masks[idx].device:
                self._param_masks[idx] = ModuleParamPruningMask._detach_tens(
                    torch.empty_like(self._params[idx].data).copy_(
                        self._param_masks[idx]
                    )
                )

            if (
                self._params_init[idx] is not None
                and self._params[idx].data.device != self._params_init[idx].device
            ):
                self._params_init[idx] = ModuleParamPruningMask._detach_tens(
                    torch.empty_like(self._params[idx].data).copy_(
                        self._params_init[idx]
                    )
                )

            if (
                self._params_unmasked[idx] is not None
                and self._params[idx].data.device != self._params_unmasked[idx].device
            ):
                self._params_unmasked[idx] = ModuleParamPruningMask._detach_tens(
                    torch.empty_like(self._params[idx].data).copy_(
                        self._params_unmasked[idx]
                    )
                )

            if (
                self._params_grad[idx] is not None
                and self._params[idx].data.device != self._params_grad[idx].device
            ):
                self._params_grad[idx] = ModuleParamPruningMask._detach_tens(
                    torch.empty_like(self._params[idx].data).copy_(
                        self._params_grad[idx]
                    )
                )

            self._scorer.check_regen_param_vals()

    def _create_hooks(self):
        for idx, (param, layer) in enumerate(zip(self._params, self._layers)):
            if self._forward_hooks[idx] is None:
                self._forward_hooks[idx] = layer.register_forward_pre_hook(
                    partial(self._hook_mask_forward, idx)
                )

            if self._allow_reintroduction and self._undo_mask_hooks[idx] is None:
                self._undo_mask_hooks[idx] = layer.register_forward_hook(
                    partial(self._hook_undo_mask, idx)
                )

            if self._gradient_hooks[idx] is None:
                self._gradient_hooks[idx] = param.register_hook(
                    partial(self._hook_mask_gradient, idx)
                )

    def _delete_hooks(self):
        if not hasattr(self, "_params"):
            return

        for idx in range(len(self._params)):
            if self._forward_hooks[idx] is not None:
                self._forward_hooks[idx].remove()
                self._forward_hooks[idx] = None

            if self._undo_mask_hooks[idx] is not None:
                self._undo_mask_hooks[idx].remove()
                self._undo_mask_hooks[idx] = None

            if self._gradient_hooks[idx] is not None:
                self._gradient_hooks[idx].remove()
                self._gradient_hooks[idx] = None

    def _hook_mask_forward(
        self, param_idx: int, mod: Module, inp: Union[Tensor, Tuple[Tensor]]
    ):
        self.apply(param_idx)

    def _hook_undo_mask(self, param_idx, module, inp, out):
        if self._allow_reintroduction:
            with torch.no_grad():
                self._params[param_idx].data.add_(self._params_unmasked[param_idx])

    def _hook_mask_gradient(self, param_idx, grad):
        if 0.0 <= self._track_grad_mom < 1.0:
            self._params_grad[param_idx].mul_(self._track_grad_mom).add_(
                (1.0 - self._track_grad_mom) * grad
            )

        return (
            grad.mul_(self._param_masks[param_idx])
            if not self._allow_reintroduction
            else grad  # do not mask gradient for movement pruning
        )

    def _setup_params_init(self):
        for idx, param in enumerate(self._params):
            if self._store_init and self._params_init[idx] is None:
                self._params_init[idx] = ModuleParamPruningMask._detach_tens(
                    param.data.clone()
                )
            elif not self._store_init and self._params_init[idx] is not None:
                self._params_init[idx] = None

    def _setup_params_unmasked(self, param_idx: int = None):
        indices = range(len(self._params)) if param_idx is None else [param_idx]

        for idx in indices:
            if self._store_unmasked and self._params_unmasked[idx] is None:
                self._params_unmasked[idx] = ModuleParamPruningMask._detach_tens(
                    self._params[idx].data.clone()
                )
            elif not self._store_unmasked and self._params_unmasked[idx] is not None:
                self._params_unmasked[idx] = None

    def _setup_params_grad(self):
        for idx, param in enumerate(self._params):
            if self._track_grad_mom >= 0.0 and self._params_grad[idx] is None:
                self._params_grad[idx] = ModuleParamPruningMask._detach_tens(
                    param.data.new_zeros(param.data.shape)
                )
            elif self._track_grad_mom < 0.0 and self._params_grad[idx] is not None:
                self._params_grad[idx] = None

    @staticmethod
    def _detach_tens(tens) -> Tensor:
        return tens.detach().requires_grad_(False)
