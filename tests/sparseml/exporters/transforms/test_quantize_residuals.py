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

from sparseml.exporters.transforms.quantize_residuals import QuantizeResiduals


@pytest.fixture
def onnx_model():
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    inp = onnx.helper.make_node("Relu", ["input"], ["relu_output"], name="relu")
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = onnx.helper.make_tensor("zero_point", onnx.TensorProto.INT8, (1,), [1])
    quant = onnx.helper.make_node(
        "QuantizeLinear",
        ["relu_output", "scale", "zero_point"],
        ["quant_output"],
        name="quant",
    )
    dequant = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_output", "scale", "zero_point"],
        ["dequant_output"],
        name="dequant",
    )
    add = onnx.helper.make_node(
        "Add", ["dequant_output", "relu_output"], ["output"], name="add"
    )

    graph = onnx.helper.make_graph(
        nodes=[inp, quant, dequant, add],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[scale, zero_point],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_relu_input(onnx_model: ModelProto):
    onnx_model = QuantizeResiduals().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [(n.name, n.op_type) for n in onnx_model.graph.node] == [
        ("relu", "Relu"),
        ("quant", "QuantizeLinear"),
        ("dequant", "DequantizeLinear"),
        ("relu_output_identity_dequantized", "DequantizeLinear"),
        ("add", "Add"),
    ]
    assert onnx_model.graph.node[-1].input == [
        "dequant_output",
        "relu_output_identity_dequantized",
    ]
    assert onnx_model.graph.node[-2].input[0] == "quant_output"


def test_add_input(onnx_model: ModelProto):
    onnx_model.graph.node[0].op_type = "Add"
    onnx_model.graph.node[0].input.append("input")
    onnx.checker.check_model(onnx_model)

    onnx_model = QuantizeResiduals().apply(onnx_model)
    onnx.checker.check_model(onnx_model)

    assert [(n.name, n.op_type) for n in onnx_model.graph.node] == [
        ("relu", "Add"),
        ("quant", "QuantizeLinear"),
        ("dequant", "DequantizeLinear"),
        ("relu_output_identity_dequantized", "DequantizeLinear"),
        ("add", "Add"),
    ]
    assert onnx_model.graph.node[-1].input == [
        "dequant_output",
        "relu_output_identity_dequantized",
    ]
    assert onnx_model.graph.node[-2].input[0] == "quant_output"
