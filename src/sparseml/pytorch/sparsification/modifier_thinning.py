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
Code to assist in the compression of structured-pruned models
"""


import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.sparsification import SparsificationTypes


__all__ = [
    "LayerThinningModifier",
    "compress_strucure_pruned_module",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class LayerThinningModifier(ScheduledModifier):
    """
    Applies layer thinning to a model based on its structurally pruned parameters,
    updating these parameters dependencies along the opposite dimensions

    | Sample yaml:
    |   !LayerThinningModifier
    |       start_epoch: 0.0
    |       update_epochs: [1.0, 2.0]
    |       param_group_dependency_map: {
    |            "param.1.name,param.2.name": ["param.1.dep.1.name"]
    |       }
    |       structure_type: filter

    :param param_group_dependency_map: mapping of comma separated parameter names that
        should be thinned together to a list of parameter names whose opposite channels
        should be updated based on which ones of the group are removed. Can be
        generated from an onnx export of the target module with
        sparseml.onnx.optim.get_param_structured_pruning_group_dependencies
        i.e. {"param.1.name,param.2.name": ["param.1.dep.1.name", "other.dep",
        "other.dep.2], ...}
    :param structure_type: type of structured pruning to apply layer thinning based on.
        pruned parameters will first be thinned along the given structure type
        dimension, and then dependencies of those parameters will be updated accordingly
        along the opposite dimension
    :param start_epoch: the epoch to apply layer thinning at
    :param update_epochs: optional list of epochs to perform an additional thinning
        step after
    :param strict: if True, all parameters in a pruning group must be sparse along
        the same indices, durring thinning will raise a ValueError if not.
        Default is True
    :param end_epoch: do not set. value forced to -1
    """

    def __init__(
        self,
        param_group_dependency_map: Dict[str, List[str]],
        structure_type: str = "filter",
        start_epoch: float = -1.0,
        update_epochs: List[float] = None,
        strict: bool = True,
        end_epoch: float = -1,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=-1,
        )

        self._param_group_dependency_map = param_group_dependency_map
        self._structure_type = structure_type
        self._update_epochs = update_epochs or []
        self._last_thinning_epoch = float("-inf")
        self._strict = strict
        self._validate()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.pruning, SparsificationTypes.structured]

    def _validate(self):
        self.validate_schedule()
        if self._structure_type not in ["filter", "channel"]:
            raise ValueError(
                f"invalid structure_type {self._structure_type}. "
                "not in ['filter', 'channel']"
            )
        if len(set(self._update_epochs)) != len(self._update_epochs):
            raise ValueError(
                f"update_epochs values may not repeat. Found {self._update_epochs}"
            )
        if list(sorted(self._update_epochs)) != self._update_epochs:
            raise ValueError(
                f"update_epochs must be sorted. Found {self._update_epochs}"
            )
        if any(
            not isinstance(key, str) for key in self._param_group_dependency_map.keys()
        ):
            non_str_keys = [
                key
                for key in self._param_group_dependency_map.keys()
                if not isinstance(key, str)
            ]
            raise ValueError(
                "keys of param_group_dependency_map must be of type str. Found keys: "
                f"{non_str_keys}"
            )

    @ModifierProp()
    def param_group_dependency_map(self) -> Dict[str, List[str]]:
        """
        :return: mapping of comma separated parameter names that should
            be thinned together to a list of parameter names whose opposite channels
            should be updated based on which ones of the group are removed
        """
        return self._param_group_dependency_map

    @ModifierProp()
    def structure_type(self) -> str:
        """
        :return: type of structured pruning to apply layer thinning based on.
            pruned parameters will first be thinned along the given structure type
            dimension, and then dependencies of those parameters will be updated
            accordingly along the opposite dimension
        """
        return self._structure_type

    @ModifierProp()
    def update_epochs(self) -> List[float]:
        """
        :return: optional list of epochs to perform an additional thinning
            step after
        """
        return self._update_epochs

    @ModifierProp()
    def strict(self) -> bool:
        """
        :return: if True, all parameters in a pruning group must be sparse along
            the same indices, durring thinning will raise a ValueError if not
        """
        return self._strict

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If this is an updatable epoch, compresses the module and any relevant optimizer
        params to match the current structured sparsity

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self._check_update_epoch(epoch):
            compress_strucure_pruned_module(
                module,
                self._param_group_dependency_map,
                self._structure_type,
                optimizer=optimizer,
                strict=self._strict,
            )

            self._last_thinning_epoch = epoch

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        return self.start_pending(epoch, steps_per_epoch) or (
            self._check_update_epoch(epoch)
        )

    def finalize(
        self, module: Optional[Module] = None, reset_loggers: bool = True, **kwargs
    ):
        """
        Cleans up any state, apply thinning if it has not yet been applied

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().finalize(module, reset_loggers, **kwargs)
        if self._last_thinning_epoch == float("-inf"):
            compress_strucure_pruned_module(
                module,
                self._param_group_dependency_map,
                self._structure_type,
                strict=self._strict,
            )

    def advance_epochs(self, ref_start_epoch: float = None):
        """
        Advance epoch attributes given a reference start epoch

        :param ref_start_epoch: the reference, i.e. new, start epoch
        """
        if ref_start_epoch is None:
            return

        super().advance_epochs(ref_start_epoch=ref_start_epoch)
        self._update_epochs = [e + ref_start_epoch for e in self._update_epochs]
        self._validate()

    def _check_update_epoch(self, epoch) -> bool:
        return any(
            self._last_thinning_epoch < update_epoch <= epoch
            for update_epoch in self._update_epochs
        )


def compress_strucure_pruned_module(
    module: Module,
    param_group_dependency_map: Dict[str, List[str]],
    structure_type: str = "filter",
    optimizer: Optional[Optimizer] = None,
    strict: bool = True,
):
    """
    Removes in-place the given module parameters along either input or output channel
    dimensions for any of those channels that are completely pruned. Compresses
    parameters grouped according to the keys in the param_group_dependency_map
    and will update the opposite channels in the dependency map to remove those same
    channels

    :param module: module to compress structurally pruned parameters of
    :param param_group_dependency_map: mapping of comma separated parameter names that
        should be pruned together to a list of parameter names whose opposite channels
        should be updated based on which ones of the group are removed.
        i.e. {("param.1.name", "param.2.name"): ["param.1.dep.1.name", "other.dep",
        "other.dep.2], ...}
    :param structure_type: type of pruning structure used to prune the model and
        generate the dependency map. Valid options are 'filter' and 'channel'.
        Default is 'filter'
    :param optimizer: optional optimizer object to update momentum buffer of for
        relevant parameters
    :param strict: if True, all parameters in a pruning group must be sparse along
        the same indices, will raise a ValueError if not. Default is True
    """
    if structure_type not in ["filter", "channel"]:
        raise ValueError(
            f"invalid structure_type {structure_type}. not in ['filter', 'channel']"
        )
    param_group_dependency_map = {
        tuple(param_group.split(",")): deps
        for param_group, deps in param_group_dependency_map.items()
    }
    named_parameters = dict(module.named_parameters())
    named_modules = dict(module.named_modules())
    param_name_to_module = {
        param_name: named_modules.get(_module_name_from_param_name(param_name))
        for param_name in named_parameters.keys()
    }

    prune_dim = 0 if structure_type == "filter" else 1  # filters stored as param 0
    for param_group, dependent_params in param_group_dependency_map.items():
        # get pruned channel idxs for each param in the group
        # then verify that they are all the same

        pruned_channel_idxs = None
        all_pruned_channel_idxs = []
        for param_name in param_group:
            if named_parameters[param_name].size(prune_dim) == 1:
                # DW Conv
                all_pruned_channel_idxs.append(None)
                continue
            pruned_idxs = _find_pruned_dims(named_parameters[param_name], prune_dim)
            all_pruned_channel_idxs.append(pruned_idxs)

            if pruned_channel_idxs is None:
                pruned_channel_idxs = pruned_idxs
                continue

            if pruned_idxs.size(0) < pruned_channel_idxs.size(0):
                # find the smallest valid pruned idx set
                pruned_idxs, pruned_channel_idxs = pruned_channel_idxs, pruned_idxs

            if pruned_channel_idxs.shape == pruned_idxs.shape and torch.all(
                pruned_channel_idxs == pruned_idxs
            ):
                continue
            elif pruned_idxs.size(0) % pruned_channel_idxs.size(0) != 0:
                raise ValueError(
                    "Incompatible size along pruning dimension for two parameters "
                    f"in the same pruning group: {pruned_idxs.size(prune_dim)} and "
                    f"{pruned_channel_idxs.size(prune_dim)}"
                )
            else:
                # check stride and equality
                stride = pruned_idxs.size(0) // pruned_channel_idxs.size(0)
                upscaled_pruned_channel_idxs = (
                    pruned_channel_idxs.reshape(-1, 1).expand(-1, stride).reshape(-1)
                )
                if strict and not torch.all(
                    upscaled_pruned_channel_idxs == pruned_idxs
                ):
                    raise ValueError(
                        "Parameters in the same pruning group have inconsistent "
                        "values pruned"
                    )
                if not upscaled_pruned_channel_idxs.numel() == pruned_idxs.numel():
                    raise ValueError(
                        "Parameters in the same pruning group have been pruned to "
                        "different structured sparsity levels"
                    )
                else:
                    continue
        if pruned_channel_idxs is None:
            _LOGGER.debug(
                f"Pruning group {param_group} found no valid pruning dimensions"
            )

        unpruned_channel_idxs = ~pruned_channel_idxs
        with torch.no_grad():
            # compress param group along pruned dimension
            for idx, param_name in enumerate(param_group):
                idxs_to_keep = (
                    unpruned_channel_idxs
                    if strict or all_pruned_channel_idxs[idx] is None
                    else ~all_pruned_channel_idxs[idx]
                )
                _compress_module_param_dim(
                    named_parameters[param_name],
                    target_dim=prune_dim,
                    idxs_to_keep=idxs_to_keep,
                    module=param_name_to_module[param_name],
                    optimizer=optimizer,
                )

            # compress dependent params along opposite dimension
            for dependent_param_name in dependent_params:
                if dependent_param_name not in named_parameters:
                    continue
                _compress_module_param_dim(
                    named_parameters[dependent_param_name],
                    target_dim=int(not prune_dim),  # 0 <-> 1
                    idxs_to_keep=unpruned_channel_idxs,
                    module=param_name_to_module[dependent_param_name],
                    optimizer=optimizer,
                )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _module_name_from_param_name(param_name: str) -> str:
    return ".".join(param_name.split(".")[:-1])


def _find_pruned_dims(param: Tensor, prune_dim: int) -> Tensor:
    # return bool tensor of size num_target_channels s.t. an element is True if all
    # values in the corresponding channel have been pruned
    num_channels = param.size(prune_dim)
    target_channel_grouped_vals = param.transpose(0, prune_dim).reshape(
        num_channels, -1
    )
    return torch.all(target_channel_grouped_vals == 0.0, dim=1)


def _compress_module_param_dim(
    param: Parameter,
    target_dim: int,
    idxs_to_keep: Tensor,
    module: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
):
    if param.dim() == 1:
        target_dim = 0

    if param.size(target_dim) == 1 and idxs_to_keep.numel() > 1:
        # DW Conv
        return

    if param.size(target_dim) % idxs_to_keep.size(0) != 0:
        _LOGGER.debug("skipping compression of parameter due to shape incompatibility")

    stride = param.data.size(target_dim) // idxs_to_keep.size(0)
    if stride > 1:
        idxs_to_keep = idxs_to_keep.reshape(-1, 1).expand(-1, stride).reshape(-1)

    param.data = (
        param.data[idxs_to_keep, ...]
        if target_dim == 0
        else param.data[:, idxs_to_keep, ...]
    )

    if param.grad is not None:
        param.grad = (
            param.grad[idxs_to_keep, ...]
            if target_dim == 0
            else param.grad[:, idxs_to_keep, ...]
        )

    if (
        optimizer is not None
        and param in optimizer.state
        and ("momentum_buffer" in optimizer.state[param])
    ):
        optimizer.state[param]["momentum_buffer"] = (
            optimizer.state[param]["momentum_buffer"][idxs_to_keep, ...]
            if target_dim == 0
            else optimizer.state[param]["momentum_buffer"][:, idxs_to_keep, ...]
        )

    # update module attrs
    if module is not None:
        # Batch Norm
        if param.dim() == 1:
            if hasattr(module, "num_features"):
                module.num_features = param.size(0)
            # BN running mean and var are not stored as Parameters so we must
            # update them here
            if hasattr(module, "running_mean") and (
                module.running_mean.size(0) == idxs_to_keep.size(0)
            ):
                module.running_mean = module.running_mean[idxs_to_keep]
            if hasattr(module, "running_var") and (
                module.running_var.size(0) == idxs_to_keep.size(0)
            ):
                module.running_var = module.running_var[idxs_to_keep]

        # Linear
        elif target_dim == 0 and hasattr(module, "out_features"):
            module.out_features = param.size(0)
        elif target_dim == 1 and hasattr(module, "in_features"):
            module.in_features = param.size(1)
        # Conv
        elif target_dim == 0 and hasattr(module, "out_channels"):
            module.out_channels = param.size(0)
        elif target_dim == 1 and hasattr(module, "in_channels"):
            module.in_channels = param.size(1)

        if (
            hasattr(module, "groups")
            and module.groups > 1
            and (hasattr(module, "out_channels") and hasattr(module, "in_channels"))
        ):
            module.groups = param.size(0) // param.size(1)
