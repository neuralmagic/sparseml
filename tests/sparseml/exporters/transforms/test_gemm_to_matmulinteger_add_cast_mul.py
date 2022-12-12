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

from sparseml.exporters.transforms.gemm_to_matmulinteger_add_cast_mul import (
    GemmToMatMulIntegerAddCastMul,
)


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
    gemm = helper.make_node(
        "Gemm",
        ["input_dequant_output", "weight_dequant_output", "bias"],
        ["gemm_output"],
        name="gemm",
    )

    graph = helper.make_graph(
        nodes=[input_dequant, weight_quant, weight_dequant, gemm],
        name="g",
        inputs=[model_input_0, model_input_1],
        outputs=[model_output],
        initializer=[x_scale, y_scale, zero_point, weight, bias],
    )
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
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
        "gemm",
    ]
    return model


def test_vanilla(onnx_model: onnx.ModelProto):
    onnx_model = GemmToMatMulIntegerAddCastMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "zero_point",
        "gemm.weight_quantized",
        "gemm_bias_add.bias_quantized",
        "gemm_bias_add.bias_quantized.scale",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "gemm_quant",
        "gemm_bias_add_quant",
        "gemm_bias_add_quant_cast",
        "gemm_bias_add_quant_rescale_mul",
    ]
    assert [n.op_type for n in onnx_model.graph.node] == [
        "MatMulInteger",
        "Add",
        "Cast",
        "Mul",
    ]


def test_gemm_no_bias_changes_nothing(onnx_model: onnx.ModelProto):
    # remove bias
    assert onnx_model.graph.initializer.pop().name == "bias"
    assert onnx_model.graph.node[-1].input.pop() == "bias"
    onnx.checker.check_model(onnx_model)

    onnx_model = GemmToMatMulIntegerAddCastMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    # NOTE: nothing changes
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "weight",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "input_dequant",
        "weight_quant",
        "weight_dequant",
        "gemm",
    ]


def test_non_default_attributes_changes_nothing(onnx_model: onnx.ModelProto):
    """
    Tests that any attributes on the gemm node make no changes occur
    """
    onnx_model.graph.node[-1].attribute.append(helper.make_attribute("alpha", 0.0))
    onnx.checker.check_model(onnx_model)

    onnx_model = GemmToMatMulIntegerAddCastMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)

    # NOTE: nothing should've changed
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "weight",
        "bias",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "input_dequant",
        "weight_quant",
        "weight_dequant",
        "gemm",
    ]
