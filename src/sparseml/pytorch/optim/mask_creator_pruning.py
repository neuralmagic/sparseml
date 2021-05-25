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
Classes for defining sparsity masks based on model parameters.
"""

import random
from abc import ABC, abstractmethod
from typing import Iterable, List, Union

import torch
from torch import Tensor


__all__ = [
    "PruningMaskCreator",
    "UnstructuredPruningMaskCreator",
    "GroupedPruningMaskCreator",
    "DimensionSparsityMaskCreator",
    "BlockPruningMaskCreator",
    "load_mask_creator",
]


class PruningMaskCreator(ABC):
    """
    Base abstract class for a sparsity mask creator.
    Subclasses should define all methods for creating masks
    """

    def create_sparsity_masks_from_tensor(self, tensors: List[Tensor]) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a masks based on their values
        :return: list of masks derived from each of the given tensors
        """
        return [torch.ne(tensor, 0.0).type(tensor.type()) for tensor in tensors]

    @abstractmethod
    def create_sparsity_masks_from_threshold(
        self, tensors: List[Tensor], threshold: Union[float, Tensor]
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a masks based on their contained
            values
        :param threshold: a threshold to determine cutoff for sparsification
        :return: list of masks derived from each of the given tensors and the threshold
        """
        raise NotImplementedError()

    @abstractmethod
    def create_sparsity_masks(
        self, tensors: List[Tensor], sparsity: float, global_sparsity: bool = False
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a masks based on their contained
            values
        :param sparsity: the desired sparsity to reach within the mask
            (decimal fraction of zeros)
        :param global_sparsity: if True, sparsity masks will be created such that the
            average sparsity across all given tensors is the target sparsity with the
            lowest global values masked. If False, each tensor will be masked to the
            target sparsity ranking values within each individual tensor. Default is
            False
        :return: list of masks (0.0 for values that are masked, 1.0 for values that are
            unmasked) calculated from the tensors such that the desired number of zeros
            matches the sparsity.
        """
        raise NotImplementedError()


class UnstructuredPruningMaskCreator(PruningMaskCreator):
    """
    Class for creating unstructured sparsity masks.
    Masks will be created using unstructured sparsity by pruning weights ranked
    by their value.  Each mask will correspond to the given tensor.
    """

    def create_sparsity_masks_from_threshold(
        self, tensors: List[Tensor], threshold: Union[float, Tensor]
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a masks based on their contained
            values
        :param threshold: a threshold at which to mask values if they are
            less than it or equal
        :return: list of masks (0.0 for values that are masked, 1.0 for values that are
            unmasked) calculated from the tensors values <= threshold are masked,
            all others are unmasked
        """
        return [(tensor > threshold).type(tensor.type()) for tensor in tensors]

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        sparsity: float,
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a mask from based on their
            contained values
        :param sparsity: the desired sparsity to reach within the mask
            (decimal fraction of zeros)
        :param global_sparsity: if True, sparsity masks will be created such that the
            average sparsity across all given tensors is the target sparsity with the
            lowest global values masked. If False, each tensor will be masked to the
            target sparsity ranking values within each individual tensor. Default is
            False
        :return: list of masks (0.0 for values that are masked, 1.0 for values that are
            unmasked) calculated from the tensors such that the desired number of zeros
            matches the sparsity.  If there are more zeros than the desired sparsity,
            zeros will be randomly chosen to match the target sparsity
        """
        if global_sparsity:
            # create tensor to make global mask with
            original_tensors = tensors
            tensors = [self._flatten_and_stack_tensors(tensors)]
        else:
            original_tensors = None

        masks = []

        for tensor in tensors:
            threshold = self._threshold_from_sparsity(tensor, sparsity)

            if threshold.numel() < 1:
                masks.append(tensor.new_ones(tensor.shape))
                continue

            min_val = tensor.min().item()
            if threshold.item() > min_val:
                masks.append((tensor > threshold).type(tensor.type()))
                continue

            # too many zeros so will go over the already given sparsity
            # and choose which zeros to not keep in mask at random
            zero_indices = (tensor == min_val).nonzero()
            rand_indices = list(range(zero_indices.shape[0]))
            random.shuffle(rand_indices)
            num_elem = tensor.numel()
            num_mask = int(num_elem * sparsity)
            rand_indices = rand_indices[:num_mask]
            rand_indices = tensor.new_tensor(rand_indices, dtype=torch.int64)
            zero_indices = zero_indices[rand_indices, :]
            mask = tensor.new_ones(tensor.shape).type(tensor.type())
            mask[zero_indices.split(1, dim=1)] = 0

            masks.append(mask.type(tensor.type()))

        if global_sparsity:
            # unpack global mask into tensor-masks with the original shapes
            global_mask = masks[0]
            masks = self._unstack_flattened_tensors(global_mask, original_tensors)
            del global_mask

        return masks

    def _threshold_from_sparsity(self, tensor: Tensor, sparsity: float) -> Tensor:
        """
        :param tensor: the tensor to find a value in for which setting
            all values < that value will give desired sparsity
        :param sparsity: the desired sparsity to apply
        :return: the threshold to get to the desired sparsity or an empty tensor
            if it was not possible given the inputs
        """
        if tensor.numel() < 1 or sparsity <= 0.0 or sparsity > 1.0:
            return tensor.new_tensor([])

        sorted_vals, _ = torch.sort(tensor.view(-1))
        lookup_index = round(sparsity * (tensor.numel() - 1))

        if lookup_index < 0:
            lookup_index = 0
        elif lookup_index > tensor.numel():
            lookup_index = tensor.numel()

        return sorted_vals[lookup_index]

    def _flatten_and_stack_tensors(self, tensors: List[Tensor]) -> Tensor:
        total_elements = sum(tensor.numel() for tensor in tensors)

        global_tensor = (
            tensors[0].new_zeros(total_elements).detach().requires_grad_(False)
        )

        curr_element = 0
        for idx, tensor in enumerate(tensors):
            global_tensor[
                curr_element : curr_element + tensor.numel()
            ] = tensor.reshape(-1)
            curr_element += tensor.numel()

        return global_tensor

    def _unstack_flattened_tensors(
        self, stacked_tensor: Tensor, original_tensors: List[Tensor]
    ) -> List[Tensor]:
        unstacked_tensors = []
        global_idx = 0
        for tensor in original_tensors:
            # unpack global tensor into masks matching original tensor shapes
            unstacked_tensor = (
                tensor.new_empty(tensor.numel()).detach().requires_grad_(False)
            )
            unstacked_tensor.copy_(
                stacked_tensor[global_idx : global_idx + tensor.numel()]
            ).type(tensor.type())
            unstacked_tensor = unstacked_tensor.reshape(tensor.shape)

            unstacked_tensors.append(unstacked_tensor)
            global_idx += tensor.numel()

        return unstacked_tensors

    def __str__(self):
        return "unstructured"

    def __repr__(self):
        return str(self)


class GroupedPruningMaskCreator(UnstructuredPruningMaskCreator):
    """
    Abstract class for a sparsity mask creator that structures masks according to
    grouping functions.  Subclasses should implement group_tensor and
    _map_mask_to_tensor
    """

    _VALID_GROUPING_FN_NAMES = ["mean", "max", "min"]

    @staticmethod
    def reduce_tensor(
        tensor: Tensor,
        dim: Union[int, List[int]],
        reduce_fn_name: str,
        keepdim: bool = True,
    ) -> Tensor:
        """

        :param tensor: the tensor to reduce
        :param dim: dimension or list of dimension to reduce along
        :param reduce_fn_name: function name to reduce tensor with. valid options
            are 'mean', 'max', 'min'
        :param keepdim: preserves the reduced dimension(s) in returned tensor shape
            as shape 1. default is True
        :return: Tensor reduced along the given dimension(s)
        """
        if reduce_fn_name == "mean":
            return torch.mean(input=tensor, dim=dim, keepdim=keepdim)
        if reduce_fn_name == "max":
            return torch.max(input=tensor, dim=dim, keepdim=keepdim)[0]
        if reduce_fn_name == "min":
            return torch.min(input=tensor, dim=dim, keepdim=keepdim)[0]
        raise ValueError(
            f"Invalid grouping fn {reduce_fn_name}, valid grouping fns: "
            f"{GroupedPruningMaskCreator._VALID_GROUPING_FN_NAMES}"
        )

    @abstractmethod
    def group_tensor(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: The tensor to reduce in groups
        :return: The grouped tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def _map_mask_to_tensor(
        self,
        grouped_mask: Tensor,
        original_tensor_shape: torch.Size,
    ) -> Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        raise NotImplementedError()

    def create_sparsity_masks_from_tensor(self, tensors: List[Tensor]) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate masks based on their values
        :return: list of masks derived from the values of the tensors grouped by
            the group_tensor function.
        """
        masks = []
        for tensor in tensors:
            grouped_tensor = self.group_tensor(tensor)
            grouped_mask = super().create_sparsity_masks_from_tensor([grouped_tensor])[
                0
            ]
            masks.append(self._map_mask_to_tensor(grouped_mask, tensor.shape))
        return masks

    def create_sparsity_masks_from_threshold(
        self, tensors: List[Tensor], threshold: Union[float, Tensor]
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate masks from based on their contained
            values
        :param threshold: a threshold of group_tensor values to determine cutoff
            for sparsification
        :return: list of masks derived from the tensors and the grouped threshold
        """
        masks = []
        for tensor in tensors:
            grouped_tensor = self.group_tensor(tensor)
            grouped_mask = super().create_sparsity_masks_from_threshold(
                [grouped_tensor], threshold
            )[0]
            masks.append(self._map_mask_to_tensor(grouped_mask, tensor.shape))
        return masks

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        sparsity: float,
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate masks from based on their contained
            values
        :param sparsity: the desired sparsity to reach within the mask
            (decimal fraction of zeros)
        :param global_sparsity: if True, sparsity masks will be created such that the
            average sparsity across all given tensors is the target sparsity with the
            lowest global values masked. If False, each tensor will be masked to the
            target sparsity ranking values within each individual tensor. Default is
            False
        :return: list of masks (0.0 for values that are masked, 1.0 for values that are
            unmasked) calculated from the tensors such that the desired number of zeros
            matches the sparsity and all values mapped to the same group have the same
            value
        """
        grouped_tensors = [self.group_tensor(tensor) for tensor in tensors]
        grouped_masks = super().create_sparsity_masks(
            grouped_tensors, sparsity, global_sparsity
        )
        masks = [
            self._map_mask_to_tensor(grouped_mask, tensor.shape)
            for grouped_mask, tensor in zip(grouped_masks, tensors)
        ]

        return masks


class DimensionSparsityMaskCreator(GroupedPruningMaskCreator):
    """
    Structured sparsity mask creator that groups sparsity blocks by the given
    dimension(s)

    :param dim: The index or list of indices of dimensions to group the mask by or
        the type of dims to prune (['channel', 'filter'])
    :param grouping_fn_name: The name of the torch grouping function to reduce
        dimensions by
    """

    def __init__(
        self,
        dim: Union[str, int, List[int]],
        grouping_fn_name: str = "mean",
    ):
        if isinstance(dim, int):
            dim = [dim]
        self._dim_name = None
        self._grouping_fn_name = grouping_fn_name
        if isinstance(dim, str):
            valid_dim_names = ["channel", "filter"]
            if dim in valid_dim_names:
                self._dim_name = dim
                self._dim = [1] if dim == "channel" else [0, 1]
            else:
                raise ValueError(
                    "Invalid Dimension name: {}, valid names: {}".format(
                        dim, valid_dim_names
                    )
                )
        self._dim = dim  # List[int]

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        sparsity: float,
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate masks from based on their contained
            values
        :param sparsity: the desired sparsity to reach within the mask
            (decimal fraction of zeros)
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
                "global_sparsity not supported for DimensionSparsityMaskCreator"
            )
        return super().create_sparsity_masks(tensors, sparsity, global_sparsity=False)

    def group_tensor(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: The tensor to transform
        :return: The mean values of the tensor grouped by the dimension(s) in self._dim
        """
        n_dims = len(tensor.shape)
        if n_dims < 3 and self._dim_name == "channel":
            raise ValueError(
                "Channel pruning not supported for fewer than 3 dimensions."
                " Received tensor with shape {}".format(tensor.shape)
            )
        reduced_dims = [idx for idx in range(n_dims) if idx not in self._dim]
        reduced_tensor = GroupedPruningMaskCreator.reduce_tensor(
            tensor, reduced_dims, self._grouping_fn_name
        )
        return reduced_tensor.type(tensor.type())

    def _map_mask_to_tensor(
        self,
        grouped_mask: Tensor,
        original_tensor_shape: torch.Size,
    ) -> Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        return grouped_mask.expand(original_tensor_shape)

    def __str__(self):
        if self._dim_name is not None:
            return self._dim_name
        return "{}:{}".format(self.__class__.__name__, self._dim)

    def __repr__(self):
        return str(self)


class BlockPruningMaskCreator(GroupedPruningMaskCreator):
    """
    Structured sparsity mask creator that groups the input tensor into blocks of
    shape block_shape.

    :param block_shape: The shape in and out channel should take in blocks.  Should be
        a list of exactly two integers that divide the input tensors evenly on the
        channel dimensions.  -1 for a dimension blocks across the entire dimension
    :param grouping_fn_name: The name of the torch grouping function to reduce
        dimensions by
    """

    def __init__(
        self,
        block_shape: List[int],
        grouping_fn_name: str = "mean",
    ):
        if len(block_shape) < 2:
            raise ValueError(
                (
                    "Invalid block_shape: {}, "
                    "block_shape must have length == 2 for in and out channels"
                ).format(block_shape)
            )

        if len(block_shape) > 2 and not all([shape == 1 for shape in block_shape[2:]]):
            # after in and out channels, only 1 can be used for other dimensions
            raise ValueError(
                (
                    "Invalid block_shape: {}, "
                    "block_shape for indices not in [0, 1] must be equal to 1"
                ).format(block_shape)
            )

        self._block_shape = block_shape
        self._grouping_fn_name = grouping_fn_name

    def group_tensor(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: The tensor to transform
        :return: The mean values of the tensor grouped by blocks of shape
            self._block_shape
        """
        blocked_tens_shape = self._get_blocked_tens_shape_and_validate(tensor.shape)
        blocked_tensor = tensor.reshape(blocked_tens_shape)
        reduced_blocks = GroupedPruningMaskCreator.reduce_tensor(
            blocked_tensor, 1, self._grouping_fn_name
        )
        return reduced_blocks.type(tensor.type())

    def _map_mask_to_tensor(
        self,
        grouped_mask: Tensor,
        original_tensor_shape: torch.Size,
    ) -> Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        blocked_tens_shape = self._get_blocked_tens_shape_and_validate(
            original_tensor_shape
        )
        # expand so every element has a corresponding value in the original tensor
        block_mask = grouped_mask.reshape(blocked_tens_shape[0], blocked_tens_shape[2])
        block_mask = block_mask.unsqueeze(1)
        block_mask = block_mask.expand(*blocked_tens_shape).contiguous()
        return block_mask.reshape(original_tensor_shape)

    def _get_blocked_tens_shape_and_validate(
        self,
        tens_shape: torch.Size,
    ) -> List[int]:
        """
        :param tens_shape: The shape of the tensor to group in blocks
        :return: shape of tens when blocked by block_shape
        :raise: ValueError if we are unable to block tens by shape block_shape
        """
        block_shape = self._block_shape
        n_dims = len(tens_shape)
        while len(block_shape) < n_dims:  # Conv will have block shape [X, Y, 1, ..., 1]
            block_shape.append(1)
        for idx, shape in enumerate(block_shape):
            if shape == -1:
                block_shape[idx] = tens_shape[idx]
        # Validate
        if n_dims < 2:
            raise ValueError(
                "Invalid tensor shape {}."
                " BlockSparsityMaskCreator can only create masks from tensors with 2 or"
                " more dimensions, tensor has {}.".format(tens_shape, n_dims)
            )
        for tens_dim, block_dim in zip(tens_shape, block_shape):
            if tens_dim % block_dim != 0:
                raise ValueError(
                    f"Invalid block_shape {block_shape} for parameter shape "
                    f"{tens_shape}. Elements of block_shape must divide parameter "
                    f"shape evenly"
                )
        # Compute blocked tensor shape
        if len(block_shape) > 1 and block_shape[1] > 1:
            return [
                tens_shape[0] * tens_shape[1] // (block_shape[0] * block_shape[1]),
                block_shape[0] * block_shape[1],
                -1,
            ]
        else:
            return [tens_shape[0] // block_shape[0], block_shape[0], -1]

    def __str__(self):
        return str(self._block_shape)

    def __repr__(self):
        return str(self)


mask_creator_name_to_constructor_lambda = {
    "unstructured": lambda: UnstructuredPruningMaskCreator(),
    "channel": lambda: DimensionSparsityMaskCreator("channel"),
    "filter": lambda: DimensionSparsityMaskCreator("filter"),
}


def load_mask_creator(obj: Union[str, Iterable[int]]) -> PruningMaskCreator:
    """
    :param obj: Formatted string or block shape iterable specifying SparsityMaskCreator
        object to return
    :return: SparsityMaskCreator object created from obj
    """
    if isinstance(obj, str) and obj in mask_creator_name_to_constructor_lambda:
        return mask_creator_name_to_constructor_lambda[obj]()
    # Checking for a BlockSparsityMaskCreator string
    if ("[" in obj and "]" in obj) or ("(" in obj and ")" in obj):
        stripped_str = obj.strip("[|]|(|)")
        block_shape = [int(s) for s in stripped_str.split(",")]
        return BlockPruningMaskCreator(block_shape)
    if isinstance(obj, list) or isinstance(obj, tuple):
        return BlockPruningMaskCreator(obj)
    raise ValueError(
        "Invalid mask type string: {}, could not map to an object".format(obj)
    )
