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
import onnx
import pytest
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.flatten_qparams import FlattenQParams
from sparsezoo.utils import validate_onnx


@pytest.fixture
def onnx_model():
    """
    Creates an ONNX model:
    ```
    input   scale  zero_point
    |       |           |
            QuantizeLinear
            |
            Output
    ```
    """
    mdl_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    mdl_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )

    zp = onnx.helper.make_tensor(
        name="zero_point", data_type=onnx.TensorProto.UINT8, dims=(1,), vals=[0]
    )
    scale = onnx.helper.make_tensor(
        name="scale", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1.0]
    )
    quantize = onnx.helper.make_node(
        "QuantizeLinear", ["input", "scale", "zero_point"], ["output"], name="id1"
    )

    graph = onnx.helper.make_graph(
        nodes=[quantize],
        name="g",
        inputs=[mdl_input],
        outputs=[mdl_output],
        initializer=[zp, scale],
    )
    model = onnx.helper.make_model(graph)
    validate_onnx(model)

    return model


def test_vanilla(onnx_model: ModelProto):
    assert len(onnx_model.graph.initializer) == 2
    assert [init.name for init in onnx_model.graph.initializer] == [
        "zero_point",
        "scale",
    ]
    zp = numpy_helper.to_array(onnx_model.graph.initializer[0])
    assert zp.shape == (1,)
    assert zp.dtype == numpy.uint8
    scale = numpy_helper.to_array(onnx_model.graph.initializer[1])
    assert scale.shape == (1,)
    assert scale.dtype == numpy.float32

    onnx_model = FlattenQParams().apply(onnx_model)

    validate_onnx(onnx_model)
    assert len(onnx_model.graph.initializer) == 2
    assert [init.name for init in onnx_model.graph.initializer] == [
        "zero_point",
        "scale",
    ]
    zp = numpy_helper.to_array(onnx_model.graph.initializer[0])
    assert zp.shape == ()
    assert zp.dtype == numpy.uint8
    scale = numpy_helper.to_array(onnx_model.graph.initializer[1])
    assert scale.shape == ()
    assert scale.dtype == numpy.float32
