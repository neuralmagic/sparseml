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
Helper functions for handling ONNX SparseTensorProto objects.
onnx >= 1.6.0 is a requirement for using sparse tensors
"""


from copy import deepcopy
from typing import Union

import numpy
from onnx import ModelProto, TensorProto, numpy_helper


try:
    from onnx import SparseTensorProto

    sparse_tensor_import_error = None
except Exception as sparse_tensor_err:
    SparseTensorProto = None
    sparse_tensor_import_error = sparse_tensor_err


__all__ = [
    "create_sparse_tensor",
    "sparse_tensor_to_dense",
    "convert_model_initializers_to_sparse",
    "convert_sparse_initializers_to_dense",
]


def _check_sparse_tensor_import():
    if sparse_tensor_import_error:
        # ONNX >= 1.6.0 required
        raise sparse_tensor_import_error


def create_sparse_tensor(
    array: Union[numpy.ndarray, TensorProto],
    name: str = None,
) -> Union[SparseTensorProto, None]:
    """
    :param array: numpy array or TensorProto object to convert to sparse representation
    :param name: name of this sparse tensor. Will be stored in
        SparseTensorProto.values.name. If the given array is a TensorProto, name will
        default to TensorProto.name
    :return: SparseTensorProto object built from the sparse representation of the input
        array
    """
    _check_sparse_tensor_import()
    if isinstance(array, TensorProto):
        if not name:
            name = array.name or None
        array = numpy_helper.to_array(array)

    # flatten array and convert to sparse
    original_dims = array.shape
    array = array.reshape(-1)
    nonzero_idxs = array.nonzero()
    nonzero_values = array[nonzero_idxs]
    nonzero_idxs = nonzero_idxs[0]  # unwrap 1-tuple
    nonzero_idxs = nonzero_idxs.astype(numpy.int64)  # required idx dtype

    # build SparseTensorProto
    return SparseTensorProto(
        values=numpy_helper.from_array(nonzero_values, name=name),
        indices=numpy_helper.from_array(nonzero_idxs),
        dims=original_dims,
    )


def sparse_tensor_to_dense(sparse_tensor: SparseTensorProto) -> TensorProto:
    """
    :param sparse_tensor: SparseTensorProto object
    :return: TensorProto object that is the dense representation of the given
        sparse tensor.
    """
    _check_sparse_tensor_import()
    name = sparse_tensor.values.name
    values = numpy_helper.to_array(sparse_tensor.values)
    indices = numpy_helper.to_array(sparse_tensor.indices)
    shape = sparse_tensor.dims

    dense_array = numpy.zeros(numpy.prod(shape)).astype(values.dtype)
    dense_array[indices] = values
    dense_array = dense_array.reshape(shape)

    return numpy_helper.from_array(dense_array, name=name)


_COMPRESSIBLE_DATA_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.INT64,
    TensorProto.INT32,
    TensorProto.INT16,
}


def convert_model_initializers_to_sparse(
    model: ModelProto, sparsity_threshold: float = 0.6, inplace: bool = True
) -> ModelProto:
    """
    :param model: ONNX model with initializers to convert to sparse
    :param sparsity_threshold: the minimum sparsity of a tensor to be converted
        to sparse representation. Default is 0.6
    :param inplace: True to do model conversion in place. Default is True
    :return: the given model with initializers above the sparsity threshold
        converted to sparse initializers
    """
    _check_sparse_tensor_import()
    if not inplace:
        model = deepcopy(model)

    sparsified_initializers = []
    for initializer in model.graph.initializer:
        if initializer.data_type not in _COMPRESSIBLE_DATA_TYPES:
            continue
        val = numpy_helper.to_array(initializer)

        sparsity = 1.0 - (numpy.count_nonzero(val) / val.size)
        if sparsity < sparsity_threshold:
            continue

        sparse_tensor = create_sparse_tensor(val, initializer.name)
        if sparse_tensor is None:
            continue

        sparsified_initializers.append(initializer)
        model.graph.sparse_initializer.append(sparse_tensor)

    for initializer in sparsified_initializers:
        model.graph.initializer.remove(initializer)

    return model


def convert_sparse_initializers_to_dense(
    model: ModelProto, inplace: bool = True
) -> ModelProto:
    """
    :param model: ONNX model with sparse initializers to convert to dense representation
    :param inplace: True to do model conversion in place. Default is True
    :return: The given model with all sparse initializers converted to dense
        initializers
    """
    _check_sparse_tensor_import()
    if not inplace:
        model = deepcopy(model)

    while model.graph.sparse_initializer:
        sparse_initializer = model.graph.sparse_initializer.pop()
        model.graph.initializer.append(sparse_tensor_to_dense(sparse_initializer))

    return model
