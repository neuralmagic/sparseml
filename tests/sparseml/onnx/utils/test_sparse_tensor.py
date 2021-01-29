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

import numpy
import pytest
from onnx import SparseTensorProto, TensorProto, numpy_helper

from sparseml.onnx.utils import create_sparse_tensor, sparse_tensor_to_dense


def _sparsify_array(array, sparsity):
    original_shape = array.shape
    array = array.reshape(-1)
    sparse_indices = numpy.random.choice(
        numpy.arange(array.size), replace=False, size=int(array.size * sparsity)
    )
    array[sparse_indices] = 0
    return array.reshape(original_shape)


def _build_random_sparse_tensor(shape, sparsity):
    array_size = numpy.prod(shape)
    num_sparse = int(array_size * sparsity)
    array = numpy.random.randn(num_sparse)
    indices = numpy.random.choice(
        numpy.arange(array_size), replace=False, size=num_sparse
    ).astype(numpy.int64)
    return SparseTensorProto(
        values=numpy_helper.from_array(array),
        indices=numpy_helper.from_array(indices),
        dims=shape,
    )


@pytest.mark.parametrize(
    "array",
    [
        (_sparsify_array(numpy.random.randn(64, 64, 3, 3), 0.7)),
        (numpy_helper.from_array(_sparsify_array(numpy.random.randn(128, 128), 0.8))),
    ],
)
def test_create_sparse_tensor(array):
    name = array.name if hasattr(array, "name") else "test"
    sparse_tensor = create_sparse_tensor(array, name=name)
    assert isinstance(sparse_tensor, SparseTensorProto)

    if isinstance(array, TensorProto):
        assert sparse_tensor.values.data_type == array.data_type
        array = numpy_helper.to_array(array)

    sparse_values = numpy_helper.to_array(sparse_tensor.values)
    sparse_indices = numpy_helper.to_array(sparse_tensor.indices)

    assert list(sparse_tensor.dims) == list(array.shape)
    assert sparse_values.size == sparse_indices.size
    assert sparse_values.dtype == array.dtype
    assert sparse_indices.dtype == numpy.int64  # required by ONNX
    assert sparse_values.size == numpy.count_nonzero(array)


@pytest.mark.parametrize(
    "sparse_tensor",
    [
        (_build_random_sparse_tensor((64, 64, 3, 3), 0.7)),
        (_build_random_sparse_tensor((128, 128), 0.8)),
    ],
)
def test_sparse_tensor_to_dense(sparse_tensor):
    tensor = sparse_tensor_to_dense(sparse_tensor)
    assert isinstance(tensor, TensorProto)
    assert tensor.data_type == sparse_tensor.values.data_type

    array = numpy_helper.to_array(tensor)
    sparse_values = numpy_helper.to_array(sparse_tensor.values)
    assert array.dtype == sparse_values.dtype
    assert list(array.shape) == list(sparse_tensor.dims)
    assert numpy.count_nonzero(array) == sparse_values.size
