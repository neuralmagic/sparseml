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
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy

from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "PruningMaskCreator",
    "UnstructuredPruningMaskCreator",
    "GroupedPruningMaskCreator",
    "DimensionPruningMaskCreator",
    "BlockPruningMaskCreator",
    "load_mask_creator",
]


class PruningMaskCreator(ABC):
    """
    Base abstract class for a sparsity mask creator.
    Subclasses should define all methods for creating masks and their initializers
    """

    @abstractmethod
    def get_mask_initializer(
        self,
        tensor: tf_compat.Tensor,
    ) -> Callable[[], tf_compat.Tensor]:
        """
        :param tensor: A tensor of a model layer's weights
        :return: Tensor initializer function for this sparsity mask
        """
        raise NotImplementedError()

    @abstractmethod
    def create_sparsity_mask(
        self,
        tensor: tf_compat.Tensor,
        sparsity: tf_compat.Tensor,
    ) -> tf_compat.Tensor:
        """
        :param tensor: A tensor of a model layer's weights
        :param sparsity: the target sparsity to use for assigning the masks
        :return: A sparsity mask close to the set sparsity based on the values of
            the input tensor
        """
        raise NotImplementedError()


class UnstructuredPruningMaskCreator(PruningMaskCreator):
    """
    Class for creating unstructured sparsity masks.
    Masks will be created using unstructured sparsity by pruning weights ranked
    by their magnitude.
    """

    def get_mask_initializer(
        self,
        tensor: tf_compat.Tensor,
    ) -> Callable[[], tf_compat.Tensor]:
        """
        :param tensor: A tensor of a model layer's weights
        :return: Initializer for tensor where an element is 1.0 for nonzero weights
         and zero for all other weights
        :raise: ValueError If the dtype is not numeric or boolean
        """

        def non_zero_mask_initializer(
            shape: tf_compat.TensorShape,
            dtype: tf_compat.DType = tf_compat.float32,
            partition_info: Any = None,  # unsued variable for compatability
        ) -> tf_compat.Tensor:
            dtype = tf_compat.as_dtype(dtype)
            if not dtype.is_numpy_compatible or dtype == tf_compat.string:
                raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)

            return tf_compat.cast(tf_compat.not_equal(tensor, 0.0), dtype=dtype)

        return non_zero_mask_initializer

    def create_sparsity_mask(
        self,
        tensor: tf_compat.Tensor,
        sparsity: tf_compat.Tensor,
    ) -> tf_compat.Tensor:
        """
        :param tensor: A tensor of a model layer's weights
        :param sparsity: the target sparsity to use for assigning the masks
        :return: A sparsity mask close to the set sparsity based on the values of
            the input tensor
        """
        abs_var = tf_compat.abs(tensor)  # Magnitudes of weights
        sparse_threshold_index = tf_compat.cast(
            tf_compat.round(
                tf_compat.cast(tf_compat.size(abs_var), tf_compat.float32) * sparsity
            ),
            tf_compat.int32,
        )
        sparse_threshold_index = tf_compat.minimum(
            tf_compat.maximum(sparse_threshold_index, 0),
            tf_compat.size(tensor) - 1,
        )

        try:
            argsort = tf_compat.argsort
        except Exception:
            try:
                argsort = tf_compat.contrib.framework.argsort
            except Exception:
                raise RuntimeError(
                    "cannot find argsort function in tensorflow_v1, "
                    "currently unsupported"
                )

        # produce tensor where each element is the index in sorted order of abs_var
        abs_var_flat = tf_compat.reshape(abs_var, [-1])
        element_ranks_flat = tf_compat.scatter_nd(
            tf_compat.expand_dims(argsort(abs_var_flat), 1),
            tf_compat.range(abs_var_flat.get_shape()[0].value),
            abs_var_flat.get_shape(),
        )
        element_ranks = tf_compat.reshape(element_ranks_flat, abs_var.get_shape())
        return tf_compat.cast(
            tf_compat.greater(element_ranks, sparse_threshold_index),
            tf_compat.float32,
        )

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

    _GROUPING_OPS = {
        "mean": tf_compat.reduce_mean,
        "max": tf_compat.reduce_max,
        "min": tf_compat.reduce_min,
    }

    @staticmethod
    def get_grouping_op(grouping_op_name: str) -> tf_compat.Operation:
        """
        :param grouping_op_name: name of grouping operation to get tf operation for
        :return: tf operation for grouping_op_name if available, raises error otherwise
        """
        if grouping_op_name not in GroupedPruningMaskCreator._GROUPING_OPS:
            raise ValueError("Invalid grouping op {}, valid grouping ops: {}").format(
                grouping_op_name, GroupedPruningMaskCreator._GROUPING_OPS
            )
        return GroupedPruningMaskCreator._GROUPING_OPS[grouping_op_name]

    @abstractmethod
    def group_tensor(self, tensor: tf_compat.Tensor) -> tf_compat.Tensor:
        """
        :param tensor: The tensor to reduce in groups
        :return: The grouped tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def _map_mask_to_tensor(
        self,
        grouped_mask: tf_compat.Tensor,
        original_tensor_shape: tf_compat.TensorShape,
    ) -> tf_compat.Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        raise NotImplementedError()

    def get_mask_initializer(
        self,
        tensor: tf_compat.Tensor,
    ) -> Callable[[], tf_compat.Tensor]:
        """
        :param tensor: A tensor of a model layer's weights
        :return: Tensor initializer function for this sparsity mask
        """

        def grouped_non_zero_mask_initializer(
            shape: tf_compat.TensorShape,
            dtype: tf_compat.DType = tf_compat.float32,
            partition_info: Any = None,  # unsued variable for compatability
        ) -> tf_compat.Tensor:
            dtype = tf_compat.as_dtype(dtype)
            if not dtype.is_numpy_compatible or dtype == tf_compat.string:
                raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)
            grouped_tensor = self.group_tensor(tensor)
            grouped_mask = tf_compat.not_equal(grouped_tensor, 0.0)
            mask = self._map_mask_to_tensor(grouped_mask, tensor.shape)
            return tf_compat.cast(mask, dtype=dtype)

        return grouped_non_zero_mask_initializer

    def create_sparsity_mask(
        self,
        tensor: tf_compat.Tensor,
        sparsity: tf_compat.Tensor,
    ) -> tf_compat.Tensor:
        """
        :param tensor: A tensor of a model layer's weights
        :param sparsity: the target sparsity to use for assigning the masks
        :return: A sparsity mask close to the set sparsity based on the values of
            the input tensor
        """
        grouped_tensor = self.group_tensor(tensor)
        grouped_mask = super().create_sparsity_mask(grouped_tensor, sparsity)
        return self._map_mask_to_tensor(grouped_mask, tensor.shape)


class DimensionPruningMaskCreator(GroupedPruningMaskCreator):
    """
    Structured sparsity mask creator that groups sparsity blocks by the given
    dimension(s)

    :param dim: The index or list of indices of dimensions to group the mask by or
        the type of dims to prune (['channel', 'filter'])
    """

    _VALID_DIM_NAMES = ["channel", "filter"]

    def __init__(
        self,
        dim: Union[str, int, List[int]],
        grouping_op_name: str = "mean",
    ):
        if isinstance(dim, int):
            dim = [dim]
        self._dim = dim  # List[int]
        self._grouping_op = GroupedPruningMaskCreator.get_grouping_op(grouping_op_name)
        self._dim_name = None
        if isinstance(dim, str):
            if dim in DimensionPruningMaskCreator._VALID_DIM_NAMES:
                self._dim_name = dim
            else:
                raise ValueError(
                    "Invalid Dimension name: {}, valid names: {}".format(
                        dim, DimensionPruningMaskCreator._VALID_DIM_NAMES
                    )
                )

    def _set_dim_by_name_for_tensor(self, tensor: tf_compat.Tensor):
        n_dims = len(tensor.shape)
        if n_dims <= 2:
            if self._dim_name == "channel":
                self._dim = [0]
            else:
                raise ValueError(
                    f"filter pruning unsupported for tensors with fewer than "
                    f"3 dimensions. Received Tensor with shape {tensor.shape}"
                )
        elif self._dim_name == "channel":
            # in channel should be the second to last dimension
            self._dim = [n_dims - 2]
        elif self._dim_name == "filter":
            # Non-kernel dimensions should be the last two in a conv (in / out channels)
            self._dim = [n_dims - 2, n_dims - 1]
        else:
            raise ValueError(
                "Invalid dimension prune type: {}, valid types: {}".format(
                    self._dim_name, DimensionPruningMaskCreator._VALID_DIM_NAMES
                )
            )

    def group_tensor(self, tensor: tf_compat.Tensor) -> tf_compat.Tensor:
        """
        :param tensor: The tensor to transform
        :return: The absolute mean values of the tensor grouped by the
            dimension(s) in self._dim
        """
        if self._dim_name is not None:
            self._set_dim_by_name_for_tensor(tensor)
        n_dims = len(tensor.shape)
        reduced_axis = [idx for idx in range(n_dims) if idx not in self._dim]
        return self._grouping_op(
            tf_compat.abs(tensor),
            axis=reduced_axis,
            keepdims=True,
        )

    def _map_mask_to_tensor(
        self,
        grouped_mask: tf_compat.Tensor,
        original_tensor_shape: tf_compat.TensorShape,
    ) -> tf_compat.Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        # using tile instead of broadcast_to for compatibility with older tf versions
        # equivalent to: tf_compat.broadcast_to(grouped_mask, original_tensor_shape)
        tile_vals = [
            dim if idx not in self._dim else 1
            for (idx, dim) in enumerate(original_tensor_shape)
        ]
        return tf_compat.tile(grouped_mask, tile_vals)

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
    block_shape must divide the shape of any input tensor evenly and must have exactly
    2 elements for the shape of in and out channels in the blocks.

    :param block_shape: The shape of blocks to strucure blocks of in and out channels
        in the mask by.  -1 represents blocking along the entire dimension.
    """

    def __init__(
        self,
        block_shape: List[int],
        grouping_op_name: str = "mean",
    ):
        if len(block_shape) != 2:
            raise ValueError(
                (
                    "Invalid block_shape: {}"
                    " , block_shape must have length == 2 for in and out channels"
                ).format(block_shape)
            )
        self._block_shape = block_shape
        self._grouping_op = GroupedPruningMaskCreator.get_grouping_op(grouping_op_name)

    def group_tensor(self, tensor: tf_compat.Tensor) -> tf_compat.Tensor:
        """
        :param tensor: The tensor to transform
        :return: The absolute mean values of the tensor grouped by blocks of
            shape self._block_shape
        """
        blocked_tens_shape, _ = self._get_blocked_tens_shape_and_validate(tensor.shape)
        # reorder so that in and out channel dimensions come before kernel
        n_dims = len(tensor.shape)
        if n_dims >= 3:
            tens_trans_dims = [n_dims - 2, n_dims - 1, *range(n_dims - 2)]
            tensor = tf_compat.transpose(tensor, tens_trans_dims)
        blocked_tens = tf_compat.reshape(tensor, blocked_tens_shape)
        reduced_blocks = self._grouping_op(
            tf_compat.abs(blocked_tens), 1, keepdims=True
        )
        return reduced_blocks

    def _map_mask_to_tensor(
        self,
        grouped_mask: tf_compat.Tensor,
        original_tensor_shape: tf_compat.TensorShape,
    ) -> tf_compat.Tensor:
        """
        :param grouped_mask: A binary mask the size of a tensor from group_tensor
        :param original_tensor_shape: Shape of the original tensor grouped_mask
            derives from
        :return: The values from grouped_mask mapped to a tensor of size
            original_tensor_shape
        """
        (
            blocked_tens_shape,
            original_tensor_shape,
        ) = self._get_blocked_tens_shape_and_validate(original_tensor_shape)
        block_values_shape = [blocked_tens_shape[0], blocked_tens_shape[2]]
        # expand so every element has a corresponding value in the original tensor
        block_mask = tf_compat.reshape(grouped_mask, block_values_shape)
        block_mask = tf_compat.expand_dims(block_mask, 1)

        # Recover reduced dimension of block_mask, using tile instead of broadcast_to
        # for compatibility with older versions of tf
        block_mask_shape = [dim.value for dim in block_mask.shape]
        tile_shape = [
            int(block_dim / mask_dim)
            for (block_dim, mask_dim) in zip(blocked_tens_shape, block_mask_shape)
        ]
        # equivalent to: tf_compat.broadcast_to(block_mask, blocked_tens_shape)
        tensor_mask_blocked = tf_compat.tile(block_mask, tile_shape)

        mask = tf_compat.reshape(tensor_mask_blocked, original_tensor_shape)
        # Undo channel / kernel transpose if applicable
        n_dims = len(original_tensor_shape)
        if n_dims >= 3:
            tens_trans_dims = [*range(2, n_dims), 0, 1]
            mask = tf_compat.transpose(mask, tens_trans_dims)
        return mask

    def _get_blocked_tens_shape_and_validate(
        self,
        tens_shape: tf_compat.TensorShape,
    ) -> Tuple[List[int], tf_compat.TensorShape]:
        """
        :param tens_shape: The shape of the tensor to group in blocks
        :return: shape of tens when blocked by block_shape and the original
            tensor shape with any transposes applied to it
        :raise: ValueError if we are unable to block tens by shape block_shape
        """
        block_shape = self._block_shape
        n_dims = len(tens_shape)
        if len(tens_shape) >= 3:  # conv should have block shape like [1, ..., 1, X, Y]
            block_shape = [*[1] * (n_dims - 2), *block_shape]
        tens_shape = [dim.value for dim in tens_shape]
        for idx, shape in enumerate(block_shape):
            if shape == -1:
                block_shape[idx] = int(tens_shape[idx])
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
        # If this is a series of conv filters, reorder so in and out channels are first
        if n_dims >= 3:
            transpose_idx = [n_dims - 2, n_dims - 1, *range(n_dims - 2)]
            block_shape = [block_shape[idx] for idx in transpose_idx]
            tens_shape = [tens_shape[idx] for idx in transpose_idx]
        # Compute blocked tensor shape
        if len(block_shape) > 1 and block_shape[1] > 1:
            blocked_tens_shape = [
                tens_shape[0] * tens_shape[1] // (block_shape[0] * block_shape[1]),
                block_shape[0] * block_shape[1],
                -1,
            ]
        else:
            blocked_tens_shape = [tens_shape[0] // block_shape[0], block_shape[0], -1]
        tens_size = numpy.prod(tens_shape)
        num_block_elements = blocked_tens_shape[0] * blocked_tens_shape[1]
        blocked_tens_shape[2] = tens_size // num_block_elements
        return blocked_tens_shape, tens_shape

    def __str__(self):
        return str(self._block_shape)

    def __repr__(self):
        return str(self)


mask_creator_name_to_constructor_lambda = {
    "unstructured": lambda: UnstructuredPruningMaskCreator(),
    "channel": lambda: DimensionPruningMaskCreator("channel"),
    "filter": lambda: DimensionPruningMaskCreator("filter"),
}


def load_mask_creator(obj: Union[str, Iterable[int]]) -> PruningMaskCreator:
    """
    :param obj: Formatted string or iterable of block_shape specifying
        SparsityMaskCreator object to return
    :return: SparsityMaskCreator object created from obj
    """
    if isinstance(obj, str) and obj in mask_creator_name_to_constructor_lambda:
        constructor_lambda = mask_creator_name_to_constructor_lambda[obj]
        return constructor_lambda()
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
