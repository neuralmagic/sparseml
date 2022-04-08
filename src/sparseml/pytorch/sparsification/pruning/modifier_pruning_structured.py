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
Modifiers and support for structured (channel/filter) pruning
Thinning (removal of pruned channels) implemented by LayerThinningModifier
"""

from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    GroupedPruningMaskCreator,
    PruningMaskCreator,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import (
    GMPruningModifier,
)
from sparseml.sparsification import GMPruningModifier as BaseGMPruningModifier
from sparseml.sparsification import SparsificationTypes
from sparseml.utils import ALL_PRUNABLE_TOKEN


__all__ = [
    "StructuredPruningMaskCreator",
    "StructuredPruningModifier",
]


class StructuredPruningMaskCreator(GroupedPruningMaskCreator):
    """
    Structured sparsity mask creator that groups sparsity blocks by the given
    dimension (channel, filter)

    :param structure_type: dimension name to prune ('channel' or 'filter')
    :param grouping_fn_name: The name of the torch grouping function to reduce
        dimensions by. Default is 'l2'
    :param tensor_group_idxs: list of lists of input tensor idxs whose given dimensions
        should be scored together. If set, all idxs in the range of provided tensors
        must be included in exactly one group (tensors in their own group should be a
        list of length 1).  If None, no tensor groups will be used
    """

    def __init__(
        self,
        structure_type: str,
        grouping_fn_name: str = "l2",
        tensor_group_idxs: Optional[List[List[int]]] = None,
    ):
        valid_structure_types = ["channel", "filter"]
        if structure_type not in valid_structure_types:
            raise ValueError(
                f"Invalid structure_type name: {structure_type}, "
                f"valid values: {valid_structure_types}"
            )

        self._structure_type = structure_type
        self._grouping_fn_name = grouping_fn_name
        self._tensor_group_idxs = tensor_group_idxs
        self._dim = [1] if structure_type == "channel" else [0]  # type: List[int]
        self._stride_grouped_tensors = {}  # Dict[int, int]: tensor_idx -> stride

    @property
    def structure_type(self) -> str:
        """
        :return: the type of structure pruned masks this mask creator produces must
            be either 'channel' or 'filter'
        """
        return self._structure_type

    @property
    def tensor_group_idxs(self) -> Optional[List[List[int]]]:
        """
        :return: list of lists of input tensor idxs whose given dimensions
              should be scored together. If set, all idxs in the range of provided
              tensors must be included in exactly one group (tensors in their own
              group should be a
        list of length 1).  If None, no tensor groups will be used
        """
        return self._tensor_group_idxs

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        target: Union[float, List[float]],
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate masks from based on their contained
            values
        :param target: the desired sparsity to reach within the mask
            (decimal fraction of zeros) can also be a list where each element is a
            sparsity for a tensor in the same position in the tensor list, If global
            sparsity is enabled, all values of the sparsity list must be the same
        :param global_sparsity: do not set True, unsupported for
            DimensionSparsityMaskCreator
        :return: list of masks (0.0 for values that are masked, 1.0 for values that are
            unmasked) calculated from the tensors such that the desired number of zeros
            matches the sparsity and all values mapped to the same group have the same
            value
        """
        if global_sparsity:
            # global sparsity unsupported because channel dims may vary across layers
            raise ValueError(
                f"global_sparsity not supported for {self.__class__.__name__}"
            )
        return super().create_sparsity_masks(tensors, target, global_sparsity=False)

    def group_tensor(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: The tensor to transform
        :return: The mean values of the tensor grouped by the dimension(s) in self._dim
        """
        n_dims = len(tensor.shape)
        reduced_dims = [idx for idx in range(n_dims) if idx not in self._dim]
        reduced_tensor = GroupedPruningMaskCreator.reduce_tensor(
            tensor, reduced_dims, self._grouping_fn_name
        )
        return reduced_tensor.type(tensor.type())

    def set_tensor_group_idxs(self, tensor_group_idxs: Optional[List[List[int]]]):
        """
        :param tensor_group_idxs: list of lists of input tensor idxs whose given
            dimensions should be scored together. If set, all idxs in the range of
            provided tensors must be included in exactly one group (tensors in their
            own group should be a list of length 1).  If None, no tensor groups will
            be used
        """
        self._tensor_group_idxs = tensor_group_idxs

    def _group_tensors(self, tensors: List[Tensor]) -> List[Tensor]:
        grouped_tensors = [self.group_tensor(tensor) for tensor in tensors]

        if self._tensor_group_idxs is None:
            return grouped_tensors

        self._validate_tensor_group_idxs(len(tensors))

        for idx_group in self._tensor_group_idxs:
            if not idx_group:
                continue

            # bypass DW Convs in group
            idx_group = [
                idx
                for idx in idx_group
                if not all(grouped_tensors[idx].size(dim) == 1 for dim in self._dim)
            ]

            if not idx_group:
                # all non-prunable along target dim
                continue

            # validate tensors have the same number of dimension elements
            group_tensors_numel = [grouped_tensors[idx].numel() for idx in idx_group]
            group_base_numel = min(group_tensors_numel)
            for group_tensor_idx, numel in zip(idx_group, group_tensors_numel):
                if numel == group_base_numel:
                    # expected number of channels
                    continue
                elif numel % group_base_numel == 0:
                    # strided conv layer
                    stride = numel // group_base_numel
                    grouped_tensor = grouped_tensors[group_tensor_idx]
                    stride_reduced_shape = [
                        dim // stride if dim != 1 else 1 for dim in grouped_tensor.shape
                    ]
                    stride_grouped_tensor = torch.zeros(
                        stride_reduced_shape,
                        dtype=grouped_tensor.dtype,
                        device=grouped_tensor.device,
                    )
                    stride_grouped_tensor.reshape(-1).copy_(
                        grouped_tensor.reshape(-1, stride).mean(axis=1).reshape(-1)
                    )
                    grouped_tensors[group_tensor_idx] = stride_grouped_tensor
                    self._stride_grouped_tensors[group_tensor_idx] = stride
                else:
                    raise ValueError(
                        "Found parameter with {numel} target dimensions in group with "
                        f"a parameter with {group_base_numel}. All parameters in a "
                        "group must have a number of elements in the target dimension "
                        "divisible by the smallest such dimension in the group"
                    )

            # calculate the group score
            tensors_in_group_flat = [
                grouped_tensors[idx].reshape(-1) for idx in idx_group
            ]
            group_score = torch.sum(torch.stack(tensors_in_group_flat), axis=0)

            # set all tensors in group to the group score
            for idx in idx_group:
                grouped_tensors[idx].view(-1).copy_(group_score)
        return grouped_tensors

    def _map_mask_to_tensor(
        self,
        grouped_mask: Tensor,
        original_tensor_shape: torch.Size,
        tensor_idx: Optional[int] = None,
    ) -> Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :param tensor_idx: optional index this tensor was passed into a tensor
            list for mask creation
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        if all(grouped_mask.size(dim) == 1 for dim in self._dim):
            # DW Conv
            return torch.ones(
                original_tensor_shape,
                dtype=grouped_mask.dtype,
                device=grouped_mask.device,
            )
        if tensor_idx and tensor_idx in self._stride_grouped_tensors:
            stride = self._stride_grouped_tensors[tensor_idx]
            unstrided_mask_shape = [
                dim * stride if dim != 1 else 1 for dim in grouped_mask.shape
            ]
            unstrided_mask = torch.zeros(
                unstrided_mask_shape,
                dtype=grouped_mask.dtype,
                device=grouped_mask.device,
            )
            unstrided_mask.reshape(-1).copy_(
                grouped_mask.reshape(-1, 1).expand(-1, stride).reshape(-1)
            )
            grouped_mask = unstrided_mask
            del self._stride_grouped_tensors[tensor_idx]

        return grouped_mask.expand(original_tensor_shape)

    def _validate_tensor_group_idxs(self, num_tensors: int):
        included_idxs = [
            idx for idx_group in self._tensor_group_idxs for idx in idx_group
        ]

        if not all(isinstance(idx, int) for idx in included_idxs):
            types = [type(idx) for idx in included_idxs]
            raise RuntimeError(
                f"all indices in tensor_group_idxs must be ints found {types}"
            )

        if len(included_idxs) != len(set(included_idxs)):
            raise RuntimeError(
                "indices may not be repeated in tensor_group_idxs. Found indices "
                f"{included_idxs}"
            )

        if len(included_idxs) != num_tensors:
            raise RuntimeError(
                f"Expected {num_tensors} indices in tensor_group_idxs for "
                f"{num_tensors}. Found {len(included_idxs)}"
            )

        expected_idxs = set(range(num_tensors))
        if any(idx not in expected_idxs for idx in included_idxs):
            raise RuntimeError(
                "tensor_group_idxs must include indices in range [0, num_tensors] "
                f" found indices: {list(sorted(included_idxs))}"
            )


@PyTorchModifierYAML()
class StructuredPruningModifier(GMPruningModifier):
    """
    Gradually applies structured kernel sparsity to a given parameter or parameters
    from init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken. Channel and filter
    pruning supported.

    A param_group_dependency_map must be provided that maps
    groups of prunable parameter names that should have their dimensions pruned
    together to a list of module parameter names that should be updated accordingly
    when those parameters are pruned.

    | Sample yaml:
    |   !StructuredPruningModifier
    |       param_groups: [
    |           ["param.1.name","param.2.name"], ["param.3.name", "param.4.name"]
    |       ]
    |       mask_type: filter
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: __ALL_PRUNABLE__
    |       leave_enabled: True
    |       inter_func: cubic

    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param param_groups: list of list of parameter names that should be pruned together
        during structured pruning so that their same indices may be removed. May be
        useful for structures such as residual blocks or grouped convolutions. Can be
        generated from an onnx export of the target module with
        sparseml.onnx.optim.get_param_structured_pruning_group_dependencies by
        splitting its comma separated keys into lists.
        i.e. [["param.1.name","param.2.name"], ["param.3.name", "param.4.name"]]
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. Defualt is __ALL_PRUNABLE__
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param mask_type: String to define type of structured sparsity (options: [
        'channel', 'filter']), or a DimensionSparsityMaskCreator object.
        default is 'filter'
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        param_groups: List[List[str]] = None,
        params: Union[str, List[str]] = ALL_PRUNABLE_TOKEN,
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "filter",
    ):
        if mask_type not in ["filter", "channel"]:
            raise ValueError(
                "StructuredPruningModifier mask_type must be 'channel' or 'filter' "
                f"found '{mask_type}'"
            )
        super(StructuredPruningModifier, self).__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            mask_type=mask_type,
        )

        self._param_groups = param_groups or []

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        # group params by idx
        param_name_to_idx = dict(zip(param_names, range(len(param_names))))
        param_group_idxs = []
        added_idxs = set()

        for param_group in self._param_groups:
            group_idxs = []
            for param_name in param_group:
                if param_name not in param_name_to_idx:
                    raise ValueError(
                        f"param {param_name} from param_groups "
                        f"not found in pruning modifier params {param_names}"
                    )
                param_idx = param_name_to_idx[param_name]
                if param_idx in added_idxs:
                    raise ValueError(
                        "found repeated param name in param_groups " f"{param_name}"
                    )
                group_idxs.append(param_idx)
                added_idxs.add(param_idx)
            param_group_idxs.append(group_idxs)
        for idx in range(len(param_names)):
            # add unadded param names
            if idx not in added_idxs:
                param_group_idxs.append([idx])

        return StructuredPruningMaskCreator(
            structure_type=self._mask_type, tensor_group_idxs=param_group_idxs
        )

    @BaseGMPruningModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.pruning, SparsificationTypes.structured]

    @ModifierProp()
    def param_groups(self) -> List[List[str]]:
        """
        :return: list of list of parameter names that should be pruned together
            during structured pruning so that their same indices may be removed. May be
            useful for structures such as residual blocks or grouped convolutions
        """
        return self._param_groups

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True for global magnitude pruning, False for
            layer-wise. [DEPRECATED] - use GlobalMagnitudePruningModifier
            for global magnitude pruning and MagnitudePruningModifier for layer-wise
        """
        return self._global_sparsity
