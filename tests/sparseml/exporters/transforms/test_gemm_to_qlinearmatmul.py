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

from sparseml.exporters.transforms.gemm_to_qlinearmatmul import GemmToQLinearMatMul


@pytest.fixture
def onnx_model() -> onnx.ModelProto:
    """
    See variant 1 in docstring of transform
    """
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
    gemm = helper.make_node(
        "Gemm",
        ["input_dequant_output", "weight_dequant_output"],
        ["gemm_output"],
        name="gemm",
    )
    output_quant = helper.make_node(
        "QuantizeLinear",
        ["gemm_output", "y_scale", "zero_point"],
        ["output_quant_output"],
        name="output_quant",
    )

    graph = helper.make_graph(
        nodes=[input_dequant, weight_quant, weight_dequant, gemm, output_quant],
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
        "gemm",
        "output_quant",
    ]
    return model


def test_vanilla(onnx_model: onnx.ModelProto):
    """
    See variant 1 in docstring of transform
    """
    onnx_model = GemmToQLinearMatMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "gemm.weight_quantized",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "gemm_quant",
    ]
    assert onnx_model.graph.node[0].op_type == "QLinearMatMul"


def test_gemm_with_bias_inserts_dequant(onnx_model: onnx.ModelProto):
    """
    See variant 2 in docstring of transform
    """
    # add in bias to gemm
    bias = helper.make_tensor("bias", onnx.TensorProto.FLOAT, (1,), [1])
    gemm = [n for n in onnx_model.graph.node if n.op_type == "Gemm"][0]
    gemm.input.append("bias")
    onnx_model.graph.initializer.append(bias)

    onnx.checker.check_model(onnx_model)

    onnx_model = GemmToQLinearMatMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "bias",
        "gemm.weight_quantized",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "gemm_quant",
        "gemm_quant_injected_dq",
        "gemm_injected_bias_add",
    ]


def test_gemm_with_bias_dequant_after(onnx_model: onnx.ModelProto):
    """
    See variant 3 in docstring of transform
    """
    # add in bias
    bias = helper.make_tensor("bias", onnx.TensorProto.FLOAT, (1,), [1])
    gemm = [n for n in onnx_model.graph.node if n.op_type == "Gemm"][0]
    gemm.input.append("bias")
    onnx_model.graph.initializer.append(bias)

    # add in dequant after the output quant node
    onnx_model.graph.node.append(
        helper.make_node(
            "DequantizeLinear",
            ["output_quant_output", "y_scale"],
            ["output_dequant_output"],
            name="output_dequant",
        )
    )
    onnx.checker.check_model(onnx_model)

    onnx_model = GemmToQLinearMatMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "bias",
        "gemm.weight_quantized",
    ]
    assert [n.name for n in onnx_model.graph.node] == [
        "gemm_quant",
        "output_dequant",
        "gemm_injected_bias_add",
    ]


def test_gemm_after_changes_nothing(onnx_model: onnx.ModelProto):
    """
    Tests that `Gemm -> QuantizeLinear -> DequantizeLinear -> Gemm`
    receives no changes
    """

    # add in dequant after the output quant node
    onnx_model.graph.node.append(
        helper.make_node(
            "DequantizeLinear",
            ["output_quant_output", "y_scale"],
            ["output_dequant_output"],
            name="output_dequant",
        )
    )
    # add in gemm after the dequant node
    onnx_model.graph.node.append(
        helper.make_node(
            "Gemm",
            ["output_dequant_output", "weight_dequant_output"],
            ["gemm2_output"],
            name="gemm2",
        )
    )
    onnx.checker.check_model(onnx_model)
    onnx_model = GemmToQLinearMatMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)

    # NOTE: nothing should've changed
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
        "output_quant",
        "output_dequant",
        "gemm2",
    ]

    # remove the gemm2 node and now things should change
    onnx_model.graph.node.pop()
    onnx.checker.check_model(onnx_model)
    onnx_model = GemmToQLinearMatMul().apply(onnx_model)
    # onnx.checker.check_model(onnx_model)

    assert [i.name for i in onnx_model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "gemm.weight_quantized",
    ]
    assert [n.name for n in onnx_model.graph.node] == ["gemm_quant", "output_dequant"]
    assert onnx_model.graph.node[0].op_type == "QLinearMatMul"


def test_non_default_attributes_changes_nothing(onnx_model: onnx.ModelProto):
    """
    Tests that any attributes on the gemm node make no changes occur
    """
    gemm_node = [n for n in onnx_model.graph.node if n.name == "gemm"][0]
    gemm_node.attribute.append(helper.make_attribute("alpha", 0.0))

    onnx_model = GemmToQLinearMatMul().apply(onnx_model)
    onnx.checker.check_model(onnx_model)

    # NOTE: nothing should've changed
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
        "output_quant",
    ]
