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

from sparseml.exporters.transforms.matmul_add_to_matmulinteger_add_cast_mul import (
    MatMulAddToMatMulIntegerAddCastMul,
)
from sparsezoo.utils import validate_onnx


@pytest.fixture
def onnx_model() -> onnx.ModelProto:
    """
    See docstring of transform
    """
    x_scale = helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, (1,), [1])
    y_scale = helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = helper.make_tensor("zero_point", onnx.TensorProto.INT8, (1,), [0])

    weight = helper.make_tensor("weight", onnx.TensorProto.FLOAT, (1,), [0])
    bias = helper.make_tensor("bias", onnx.TensorProto.FLOAT, (1,), [0])
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
    transpose = helper.make_node(
        "Transpose", ["weight_dequant_output"], ["transpose_output"], name="transpose"
    )
    matmul = helper.make_node(
        "MatMul",
        ["input_dequant_output", "transpose_output"],
        ["matmul_output"],
        name="matmul",
    )
    add = helper.make_node("Add", ["matmul_output", "bias"], ["output"], name="add")

    graph = helper.make_graph(
        nodes=[input_dequant, weight_quant, weight_dequant, transpose, matmul, add],
        name="g",
        inputs=[model_input_0, model_input_1],
        outputs=[model_output],
        initializer=[x_scale, y_scale, zero_point, weight, bias],
    )
    model = helper.make_model(graph)
    validate_onnx(model)
    assert [i.name for i in model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "weight",
        "bias",
    ]
    assert [n.name for n in model.graph.node] == [
        "input_dequant",
        "weight_quant",
        "weight_dequant",
        "transpose",
        "matmul",
        "add",
    ]
    return model


def test_vanilla(onnx_model: onnx.ModelProto):
    onnx_model = MatMulAddToMatMulIntegerAddCastMul().apply(onnx_model)
    validate_onnx(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "zero_point",
        "matmul.weight_quantized",
        "add.bias_quantized",
        "add.bias_quantized.scale",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "matmul_quant",
        "matmul_bias_add_quant",
        "matmul_bias_add_quant_cast",
        "matmul_bias_add_quant_rescale_mul",
    ]
    assert [n.op_type for n in onnx_model.graph.node] == [
        "MatMulInteger",
        "Add",
        "Cast",
        "Mul",
    ]


def test_without_transpose(onnx_model: onnx.ModelProto):
    onnx_model.graph.node[-2].input[1] = "weight_dequant_output"
    onnx_model.graph.node.pop(-3)
    validate_onnx(onnx_model)
    onnx_model = MatMulAddToMatMulIntegerAddCastMul().apply(onnx_model)
    validate_onnx(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "zero_point",
        "matmul.weight_quantized",
        "add.bias_quantized",
        "add.bias_quantized.scale",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "matmul_quant",
        "matmul_bias_add_quant",
        "matmul_bias_add_quant_cast",
        "matmul_bias_add_quant_rescale_mul",
    ]
    assert [n.op_type for n in onnx_model.graph.node] == [
        "MatMulInteger",
        "Add",
        "Cast",
        "Mul",
    ]


def test_matmul_no_bias_converts(onnx_model: onnx.ModelProto):
    # remove "bias" initializer and "add" node
    assert onnx_model.graph.initializer.pop().name == "bias"
    assert onnx_model.graph.node.pop().name == "add"
    onnx_model.graph.output[0].name = "matmul_output"  # update graph output name
    validate_onnx(onnx_model)

    onnx_model = MatMulAddToMatMulIntegerAddCastMul().apply(onnx_model)
    validate_onnx(onnx_model)
    # converted model should have matmulinteger + rescale mul without bias add
    assert [i.name for i in onnx_model.graph.initializer] == [
        "zero_point",
        "matmul.weight_quantized",
        "matmul_quant.rescale.scale",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "matmul_quant",
        "matmul_bias_add_quant_cast",
        "matmul_quant_rescale_mul",
    ]
