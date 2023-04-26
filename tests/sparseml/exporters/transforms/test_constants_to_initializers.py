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

import onnx
import pytest
from onnx import ModelProto

from sparseml.exporters.transforms.constants_to_initializers import (
    ConstantsToInitializers,
)
from sparsezoo.utils import validate_onnx


@pytest.fixture
def onnx_model():
    """
    | Creates an ONNX model:
    |   INPUT   Constant
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
    constant = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const"],
        name="const_node",
        value=onnx.helper.make_tensor(
            name="const_value", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
        ),
    )
    add = onnx.helper.make_node("Add", ["input", "const"], ["output"], name="add")

    graph = onnx.helper.make_graph(
        nodes=[constant, add],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[],
    )

    model = onnx.helper.make_model(graph)
    validate_onnx(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = ConstantsToInitializers().apply(onnx_model)
    validate_onnx(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["add"]
    assert [i.name for i in onnx_model.graph.initializer] == ["const"]
