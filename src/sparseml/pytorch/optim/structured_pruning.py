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


__all__ = ["compress_strucure_pruned_module"]


_LOGGER = logging.getLogger(__name__)


def compress_strucure_pruned_module(
    module: Module,
    param_group_dependency_map: Dict[str, List[str]],
    structure_type: str = "filter",
    optimizer: Optional[Optimizer] = None,
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
        for param_name in param_group:
            if named_parameters[param_name].size(prune_dim) == 1:
                # DW Conv
                continue
            pruned_idxs = _find_pruned_dims(named_parameters[param_name], prune_dim)

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
                if not torch.all(upscaled_pruned_channel_idxs == pruned_idxs):
                    raise ValueError(
                        "Parameters in the same pruning group have inconsistent "
                        "values pruned"
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
            for param_name in param_group:
                _compress_module_param_dim(
                    named_parameters[param_name],
                    target_dim=prune_dim,
                    idxs_to_keep=unpruned_channel_idxs,
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
