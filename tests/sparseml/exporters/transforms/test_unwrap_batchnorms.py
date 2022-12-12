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

from sparseml.exporters.transforms.unwrap_batchnorms import UnwrapBatchNorms


@pytest.fixture
def onnx_model():
    """
    | Creates an ONNX model:
    |   INPUT   init1
    |       |    |
    |        ADD
    |         |
    |       OUTPUT
    """
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    init1 = numpy_helper.from_array(
        numpy.array([-128, -64, 0, 64, 127], dtype=numpy.int8),
        "init1.bn_wrapper_replace_me",
    )
    add = onnx.helper.make_node(
        "Add", ["input", "init1.bn_wrapper_replace_me"], ["output"], name="add"
    )

    graph = onnx.helper.make_graph(
        nodes=[add],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[init1],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    assert [n.name for n in onnx_model.graph.node] == ["add"]
    assert [i.name for i in onnx_model.graph.initializer] == [
        "init1.bn_wrapper_replace_me"
    ]
    assert onnx_model.graph.node[0].input == ["input", "init1.bn_wrapper_replace_me"]
    assert onnx_model.graph.node[0].output == ["output"]

    onnx_model = UnwrapBatchNorms().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["add"]
    assert [i.name for i in onnx_model.graph.initializer] == ["init1"]
    assert onnx_model.graph.node[0].input == ["input", "init1"]
    assert onnx_model.graph.node[0].output == ["output"]
