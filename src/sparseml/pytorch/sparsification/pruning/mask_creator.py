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
Base classes for defining sparsity masks based on model parameters. Includes
implementations for commonly used mask creators in the sparseml ecosystem
including unstructured and four block
"""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Union

import torch
from torch import Tensor

from sparseml.pytorch.utils import memory_aware_threshold


__all__ = [
    "PruningMaskCreator",
    "GroupedPruningMaskCreator",
    "UnstructuredPruningMaskCreator",
    "FourBlockMaskCreator",
    "BlockMaskCreator",
    "NMPruningMaskCreator",
]


class PruningMaskCreator(ABC):
    """
    Base abstract class for a sparsity mask creator.
    Subclasses should define all methods for creating masks
    """

    @abstractmethod
    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        target: Union[float, List[float]],
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a masks based on their contained
            values
        :param target: the desired sparsity (decimal fraction of zeros) to reach
            within the mask or other float target value to base sparsity masks on.
            Can also be a list where each element is a
            target for a tensor in the same position in the tensor list. If global
            sparsity is enabled, all values of the target list must be the same
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

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        target: Union[float, List[float]],
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a mask from based on their
            contained values
        :param target: the desired sparsity (decimal fraction of zeros) to reach
            within the mask. Can also be a list where each element is a
            target for a tensor in the same position in the tensor list. If global
            sparsity is enabled, all values of the target list must be the same
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
        sparsity = target  # target should be desired sparsity level
        if isinstance(sparsity, float):
            sparsity = [sparsity] * len(tensors)
        if len(sparsity) != len(tensors):
            raise ValueError(
                "a sparsity target must be defined for every given Tensor. Received"
                f"{len(sparsity)} targets for {len(tensors)} Tensors."
            )

        if global_sparsity:
            # create tensor to make global mask with
            original_tensors = tensors
            tensors = [self._flatten_and_stack_tensors(tensors)]
            if not all(target == sparsity[0] for target in sparsity):
                raise ValueError(
                    "all sparsity targets must be the same for global pruning "
                    f"received targets: {sparsity}"
                )
            sparsity = [sparsity[0]]
        else:
            original_tensors = None

        masks = []

        for tensor, sparsity_target in zip(tensors, sparsity):
            threshold = self._threshold_from_sparsity(tensor, sparsity_target)

            if threshold.numel() < 1:
                masks.append(tensor.new_ones(tensor.shape))
                continue

            num_elem = tensor.numel()
            target_num_mask = round(num_elem * sparsity_target)
            min_val = tensor.min().item()

            if threshold.item() > min_val:
                threshold_mask = tensor > threshold

                num_masked = num_elem - torch.sum(threshold_mask).item()
                if num_masked != target_num_mask:
                    # attempt to reconcile expected number of masked weights
                    # may occur if multiple values have the threshold weight
                    num_to_flip = abs(num_masked - target_num_mask)
                    over_masked = num_masked > target_num_mask
                    threshold_mask = self._flip_threshold_mask_vals(
                        threshold_mask, tensor, threshold, num_to_flip, over_masked
                    )

                masks.append(threshold_mask.type(tensor.type()))
                continue

            # too many zeros so will go over the already given sparsity
            # and choose which zeros to not keep in mask at random
            zero_indices = (tensor == min_val).nonzero(as_tuple=False)
            rand_indices = list(range(zero_indices.shape[0]))
            local_rng = random.Random(42)
            local_rng.shuffle(rand_indices)
            rand_indices = rand_indices[:target_num_mask]
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
        :param sparsity: the desired sparsity to reach within the mask
            (decimal fraction of zeros) can also be a list where each element is a
            sparsity for a tensor in the same position in the tensor list
        :return: the threshold to get to the desired sparsity or an empty tensor
            if it was not possible given the inputs
        """
        if tensor.numel() < 1 or sparsity <= 0.0 or sparsity > 1.0:
            return tensor.new_tensor([])

        lookup_index = round(sparsity * tensor.numel()) - 1
        if lookup_index < 0:
            lookup_index = 0
        elif lookup_index > tensor.numel():
            lookup_index = tensor.numel()

        return memory_aware_threshold(tensor, lookup_index)

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

    def _flip_threshold_mask_vals(
        self,
        mask: Tensor,
        tensor: Tensor,
        threshold: Tensor,
        max_flip: int,
        over_masked: bool,
    ) -> Tensor:
        # flip mask values where tensor == threshold until mask has desired
        # number of 0s/1s
        threshold_idxs = torch.nonzero(tensor == threshold, as_tuple=False)
        num_flipped = 0
        for threshold_elem_idx in threshold_idxs:
            # make tensor returned by nonzero() indexable
            threshold_elem_idx = threshold_elem_idx.split(1)
            threshold_mask_elem = mask[threshold_elem_idx]

            # flip mask val at threshold index if necessary
            if over_masked and threshold_mask_elem == 0:
                mask[threshold_elem_idx] = 1
                num_flipped += 1
            elif not over_masked and threshold_mask_elem == 1:
                mask[threshold_elem_idx] = 0
                num_flipped += 1

            if num_flipped >= max_flip:
                break
        return mask


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
            are 'l2', 'mean', 'max', 'min'
        :param keepdim: preserves the reduced dimension(s) in returned tensor shape
            as shape 1. default is True
        :return: Tensor reduced along the given dimension(s)
        """
        reduce_fn_name = reduce_fn_name.lower()
        if reduce_fn_name == "l2":
            return torch.linalg.vector_norm(input=tensor, dim=dim, keepdim=keepdim)
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
        raise NotImplementedError()

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        target: Union[float, List[float]],
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate masks from based on their contained
            values
        :param target: the desired sparsity (decimal fraction of zeros) to reach
            within the mask. Can also be a list where each element is a
            target for a tensor in the same position in the tensor list. If global
            sparsity is enabled, all values of the target list must be the same
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
        sparsity = target
        grouped_tensors = self._group_tensors(tensors)
        grouped_masks = super().create_sparsity_masks(
            grouped_tensors, sparsity, global_sparsity
        )
        masks = [
            self._map_mask_to_tensor(grouped_mask, tensor.shape, idx)
            for idx, (grouped_mask, tensor) in enumerate(zip(grouped_masks, tensors))
        ]

        return masks

    def _group_tensors(self, tensors: List[Tensor]) -> List[Tensor]:
        return [self.group_tensor(tensor) for tensor in tensors]


class FourBlockMaskCreator(GroupedPruningMaskCreator):
    """
    semi-structured sparsity mask creator that groups sparsity blocks in groups of four
    along the input-channel dimension (assumed to be dimension 1 for pytorch)

    Equivalent to BlockPruningMaskCreator([1, 4]) without restrictions on number
    of dimensions, or divisibility

    :param grouping_fn_name: The name of the torch grouping function to reduce
        dimensions by
    """

    def __init__(
        self,
        grouping_fn_name: str = "mean",
    ):
        self._grouping_fn_name = grouping_fn_name

    def group_tensor(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: The tensor to transform
        :return: The mean values of the tensor grouped by blocks of shape
            self._block_shape
        """
        if tensor.dim() > 2:
            # permute input channel dim to last dimension
            permute_val = list(range(tensor.dim()))
            del permute_val[1]
            permute_val.append(1)
            tensor = tensor.permute(*permute_val)

        remainder = tensor.size(-1) % 4
        if remainder != 0:
            # pad with zeros to make masks add to 4
            pad_num = 4 - remainder
            padded_tensor = torch.zeros(
                *tensor.shape[:-1],
                tensor.size(-1) + pad_num,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            padded_tensor[..., :-pad_num] = tensor
            padded_tensor[..., -pad_num:] = torch.mean(
                # mean of remainder input channel dims
                tensor[..., -remainder:],
                dim=-1,
                keepdim=True,
            )
            tensor = padded_tensor

        blocked_tensor = tensor.reshape(-1, 4)
        reduced_blocks = GroupedPruningMaskCreator.reduce_tensor(
            blocked_tensor, 1, self._grouping_fn_name
        )
        return reduced_blocks.type(tensor.type())

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
        # expand so every element has a corresponding value in the original tensor
        block_mask = grouped_mask.expand(-1, 4).contiguous()

        # adjust for permuted shape if necessary
        original_tensor_shape = list(original_tensor_shape)
        if len(original_tensor_shape) > 2:
            original_tensor_shape.append(original_tensor_shape[1])
            del original_tensor_shape[1]

        # adjust for padding if necessary
        remainder = original_tensor_shape[-1] % 4
        if remainder != 0:
            original_tensor_shape[-1] += 4 - remainder

        # set to original shape
        block_mask = block_mask.reshape(original_tensor_shape)

        # remove padding if necessary
        if remainder != 0:
            pad_num = 4 - remainder
            block_mask = block_mask[..., :-pad_num]

        # repermute mask if necessary
        if len(original_tensor_shape) > 2:
            permute_val = list(range(len(original_tensor_shape)))
            del permute_val[-1]
            permute_val.insert(1, len(permute_val))
            block_mask = block_mask.permute(*permute_val)
        return block_mask


class BlockMaskCreator(GroupedPruningMaskCreator):
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

        self._block_shape = deepcopy(block_shape)
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


class NMPruningMaskCreator(PruningMaskCreator):
    """
    Class for creating N:M sparsity masks.
    Masks will be created using the N:M ratio, where for every block of M weights,
    N will be pruned based on ranked weight value. Each mask will correspond to the
    given tensor.

    :param N: The number of weights in a group to keep
    :param M: The size of a weight group
    """

    def __init__(
        self,
        N: int = 2,
        M: int = 4,
    ):
        self._N = N
        self._M = M

    def create_sparsity_masks(
        self,
        tensors: List[Tensor],
        target: Union[float, List[float]],
        global_sparsity: bool = False,
    ) -> List[Tensor]:
        """
        :param tensors: list of tensors to calculate a masks based on their contained
            values
        :param target: the desired sparsity (decimal fraction of zeros) to reach
            within the mask or other float target value to base sparsity masks on.
            Can also be a list where each element is a target for a tensor in the same
            position in the tensor list. The target value must be within 1e-2 of the
            effective sparsity of the N:M ratio.
        :param global_sparsity: typically used to determine pruning masks globally.
            Not used here because global sparsity doesn't apply to N:M pruning
        :return: list of masks (0.0 for values that are masked, 1.0 for values that are
            unmasked) calculated from the tensors such that the desired number of zeros
            matches the sparsity.
        """
        nm_sparsity = 1 - (self._N / self._M)
        if (isinstance(target, float) and target == 0.0) or (
            isinstance(target, List) and all([sparsity == 0.0 for sparsity in target])
        ):
            return [torch.ones_like(tensor) for tensor in tensors]
        if (isinstance(target, float) and abs(nm_sparsity - target) > 1e-2) or (
            isinstance(target, List)
            and any([abs(nm_sparsity - sparsity) > 1e-2 for sparsity in target])
        ):
            raise ValueError(
                "Sparsity must match N:M ratio. e.g. if using '3:4' then sparsity "
                "should be set to 0.25"
            )

        masks = []
        for tensor in tensors:
            if tensor.numel() % self._M != 0:
                raise ValueError(
                    f"Tensor of size {tensor.shape} can't be evenly divided into "
                    f"{self._M} groups"
                )
            original_tensor = tensor.clone()
            num_groups = tensor.numel() // self._M
            if len(tensor.shape) == 4:
                # N:M sparsity for convolutional layers
                tensor_temp = (
                    tensor.detach()
                    .abs()
                    .permute(0, 2, 3, 1)
                    .reshape(num_groups, self._M)
                )
                index = torch.argsort(tensor_temp, dim=1)[:, : int(self._M - self._N)]
                w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
                masks.append(
                    w_b.scatter_(dim=1, index=index, value=0).reshape(
                        original_tensor.permute(0, 2, 3, 1).shape
                    )
                )
            elif len(tensor.shape) == 2:
                # N:M sparsity for linear layers
                tensor_temp = tensor.detach().abs().reshape(num_groups, self._M)
                index = torch.argsort(tensor_temp, dim=1)[:, : int(self._M - self._N)]
                w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
                masks.append(
                    w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)
                )
            else:
                raise NotImplementedError("Only support layers of dimension 2 or 4")
        return masks


def get_mask_creator_default(mask_type: Union[str, List[int]]) -> PruningMaskCreator:
    """
    :param mask_type: type of mask creator to use, can be 'unstructured', for
        unstructured mask creator, 'block4' for 1x4 block pruning, 'N:M' where N and M
        are integers for N:M pruning, or a list of two integers for custom block
        pruning (does not support padding)
    :return: mask creator object created from the mask type
    """
    if mask_type == "unstructured":
        return UnstructuredPruningMaskCreator()
    elif mask_type == "block4":
        return FourBlockMaskCreator()
    elif ":" in mask_type:
        nm = mask_type.split(":")
        if len(nm) != 2:
            raise ValueError(
                "N:M pruning must be specified in the format 'N:M' with "
                f"2 values, but {len(nm)} values were found"
            )
        return NMPruningMaskCreator(N=int(nm[0]), M=int(nm[1]))
    elif mask_type == "tensorrt":
        return NMPruningMaskCreator(N=2, M=4)
    elif isinstance(mask_type, List):
        if not all(isinstance(val, int) for val in mask_type):
            raise ValueError(
                "all values in list specification of BlockMaskCreator must be integers "
                f"found {mask_type}"
            )
        if len(mask_type) != 2:
            raise ValueError(
                "expected list of length 2 for specification of BlockMaskCreator, "
                f"got list with length {len(mask_type)}, mask_type={mask_type}"
            )
        return BlockMaskCreator(mask_type)
    else:
        raise ValueError(
            f"Unknown mask_type {mask_type}. Supported mask types include "
            "'unstructured' and 'block4'"
        )
