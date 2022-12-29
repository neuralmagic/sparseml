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
from onnx import helper

from sparseml.exporters.transforms.conv_to_qlinearconv import ConvToQLinearConv


@pytest.fixture
def onnx_model() -> onnx.ModelProto:
    """See graph structure described by ConvToQLinearConv"""
    x_scale = helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, (1,), [1])
    y_scale = helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = helper.make_tensor("zero_point", onnx.TensorProto.INT8, (1,), [0])

    weight_init = helper.make_tensor("weight", onnx.TensorProto.FLOAT, (1,), [0])
    model_input_0 = helper.make_tensor_value_info(
        "input_0", onnx.TensorProto.FLOAT, (1,)
    )
    model_input_1 = helper.make_tensor_value_info(
        "input_1", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (1,))

    input_dequant = helper.make_node(
        "DequantizeLinear",
        ["input_0", "x_scale", "zero_point"],
        ["input_dequant_output"],
        name="input_dequant",
    )
    weight_quant = helper.make_node(
        "QuantizeLinear",
        ["weight", "y_scale", "zero_point"],
        ["weight_quant_output"],
        name="weight_quant",
    )
    weight_dequant = helper.make_node(
        "DequantizeLinear",
        ["weight_quant_output", "x_scale", "zero_point"],
        ["weight_dequant_output"],
        name="weight_dequant",
    )
    conv = helper.make_node(
        "Conv",
        ["input_dequant_output", "weight_dequant_output"],
        ["conv_output"],
        name="conv",
    )
    output_quant = helper.make_node(
        "QuantizeLinear",
        ["conv_output", "y_scale", "zero_point"],
        ["output_quant_output"],
        name="output_quant",
    )

    graph = helper.make_graph(
        nodes=[input_dequant, weight_quant, weight_dequant, conv, output_quant],
        name="g",
        inputs=[model_input_0, model_input_1],
        outputs=[model_output],
        initializer=[x_scale, y_scale, zero_point, weight_init],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "weight",
    ]
    assert [n.name for n in model.graph.node] == [
        "input_dequant",
        "weight_quant",
        "weight_dequant",
        "conv",
        "output_quant",
    ]
    return model


def test_vanilla(onnx_model: onnx.ModelProto):
    onnx_model = ConvToQLinearConv().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "conv.weight_quantized",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "conv_quant",
    ]
    assert onnx_model.graph.node[0].op_type == "QLinearConv"


def test_with_bias(onnx_model: onnx.ModelProto):
    bias = helper.make_tensor("bias", onnx.TensorProto.FLOAT, (1,), [1])
    gemm = [n for n in onnx_model.graph.node if n.op_type == "Conv"][0]
    gemm.input.append("bias")
    onnx_model.graph.initializer.append(bias)
    onnx.checker.check_model(onnx_model)

    onnx_model = ConvToQLinearConv().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "conv.weight_quantized",
        "conv.bias_quantized",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "conv_quant",
    ]
    assert onnx_model.graph.node[0].op_type == "QLinearConv"
