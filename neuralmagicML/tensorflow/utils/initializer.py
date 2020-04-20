"""
Initializer functions for Tensors.
"""

from typing import Any

from neuralmagicML.tensorflow.utils import (
    tf_compat,
)


__all__ = ["non_zero_mask_initializer"]


def non_zero_mask_initializer(weights: tf_compat.Tensor):
    """
    :param weights: A tensor of a model layer's weights
    :return: Initializer for tensor where an element is 1.0 for nonzero weights
     and zero for all other weights
    :raise: ValueError If the dtype is not numeric or boolean
    """

    def non_zero_mask_initializer(
        shape: tf_compat.TensorShape,
        dtype: tf_compat.dtypes.DType = tf_compat.dtypes.float32,
        partition_info: Any = None,  # unsued variable for compatability
    ):
        dtype = tf_compat.dtypes.as_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == tf_compat.dtypes.string:
            raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)
        return tf_compat.cast(tf_compat.math.not_equal(weights, 0.0), dtype=dtype)

    return non_zero_mask_initializer
