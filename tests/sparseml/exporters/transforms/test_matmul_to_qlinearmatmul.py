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

from sparseml.exporters.transforms.matmul_to_qlinearmatmul import MatMulToQLinearMatMul
from tests.sparseml.exporters.transforms.test_onnx_transform import (
    _create_model as _create_model_no_matmul,
)


def _create_test_model(with_transpose=False, with_reshape=False):
    """
    Creates a test model with a matmul node and quantize/dequantize nodes

    |          INPUT_0           INPUT_1
    |            |               |
    |     QuantizeLinear     QuantizeLinear
    |            |               |
    |     DequantizeLinear   DequantizeLinear
    |                  |      |
    |                   MatMul
    |                     |
    |                  Transpose (optional, if `with_transpose` is True)
    |                     |
    |                  Reshape (optional, if `with_reshape` is True)
    |                     |
    |               QuantizeLinear
    |                     |
    |              DequantizeLinear
    |                     |
    |                  OUTPUT
    """
    x_scale = onnx.helper.make_tensor(
        "x_scale",
        onnx.TensorProto.FLOAT,
        (1,),
        [1],
    )
    y_scale = onnx.helper.make_tensor(
        "y_scale",
        onnx.TensorProto.FLOAT,
        (1,),
        [1],
    )
    zero_point = onnx.helper.make_tensor(
        "zero_point",
        onnx.TensorProto.INT8,
        (1,),
        [1],
    )

    model_input_0 = onnx.helper.make_tensor_value_info(
        "input_0", onnx.TensorProto.FLOAT, (1,)
    )
    model_input_1 = onnx.helper.make_tensor_value_info(
        "input_1", onnx.TensorProto.FLOAT, (1,)
    )

    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )

    quantize_linear_node_0 = onnx.helper.make_node(
        "QuantizeLinear",
        ["input_0", "y_scale", "zero_point"],
        ["quant_linear_0_output"],
        name="quantize_linear_node_0",
    )
    dequantize_linear_node_0 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_0_output", "x_scale", "zero_point"],
        ["dequant_linear_0_output"],
        name="dequantize_linear_node_0",
    )

    quantize_linear_node_1 = onnx.helper.make_node(
        "QuantizeLinear",
        ["input_1", "y_scale"],
        ["quant_linear_1_output"],
        name="quantize_linear_node_1",
    )
    dequantize_linear_node_1 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_1_output", "x_scale", "zero_point"],
        ["dequant_linear_1_output"],
        name="dequantize_linear_node_1",
    )

    matmul_node = onnx.helper.make_node(
        "MatMul",
        ["dequant_linear_0_output", "dequant_linear_1_output"],
        ["matmul_output"],
        name="matmul_node",
    )

    quantize_linear_node_2 = onnx.helper.make_node(
        "QuantizeLinear",
        ["matmul_output", "y_scale", "zero_point"],
        ["quant_linear_2_output"],
        name="quantize_linear_node_2",
    )
    dequantize_linear_node_2 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_2_output", "x_scale", "zero_point"],
        ["dequant_linear_2_output"],
        name="dequantize_linear_node_2",
    )

    graph = onnx.helper.make_graph(
        nodes=[
            quantize_linear_node_0,
            dequantize_linear_node_0,
            quantize_linear_node_1,
            dequantize_linear_node_1,
            matmul_node,
            quantize_linear_node_2,
            dequantize_linear_node_2,
        ],
        name="g",
        inputs=[model_input_0, model_input_1],
        outputs=[model_output],
        initializer=[x_scale, y_scale, zero_point],
    )

    if with_transpose and not with_reshape:
        graph = _add_transpose_node(graph)

    if with_reshape and not with_transpose:
        graph = _add_reshape_node(graph)

    if with_reshape and with_transpose:
        graph = _add_reshape_and_transpose_node(graph)

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def _add_transpose_node(graph):
    transpose_node = onnx.helper.make_node(
        "Transpose", ["matmul_output"], ["transpose_output"], name="transpose_node"
    )
    graph.node.insert(5, transpose_node)
    graph.node[6].input[0] = "transpose_output"
    return graph


def _add_reshape_node(graph):
    shape = onnx.helper.make_tensor("reshape", onnx.TensorProto.INT64, (1,), [1])
    reshape_node = onnx.helper.make_node(
        "Reshape",
        ["matmul_output", "reshape"],
        ["reshape_output"],
        name="reshape_node",
    )
    graph.node.insert(5, reshape_node)
    graph.node[6].input[0] = "reshape_output"
    graph.initializer.append(shape)
    return graph


def _add_reshape_and_transpose_node(graph):
    shape = onnx.helper.make_tensor("reshape", onnx.TensorProto.INT64, (1,), [1])

    transpose_node = onnx.helper.make_node(
        "Transpose", ["matmul_output"], ["transpose_output"], name="transpose_node"
    )
    reshape_node = onnx.helper.make_node(
        "Reshape",
        ["transpose_output", "reshape"],
        ["reshape_output"],
        name="reshape_node",
    )

    graph.node.insert(5, reshape_node)
    graph.node.insert(5, transpose_node)
    graph.node[7].input[0] = "reshape_output"
    graph.initializer.append(shape)
    return graph


"""
Testing functions for every use case
"""


def _test_matmul(model):
    assert [node.name for node in model.graph.node] == [
        "quantize_linear_node_0",
        "quantize_linear_node_1",
        "matmul_node_quant",
        "dequantize_linear_node_2",
    ]
    assert [node.name for node in model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
    ]
    assert model.graph.input[0].name == "input_0"
    assert model.graph.output[0].name == "output"


def _test_transpose(model):
    assert [node.name for node in model.graph.node] == [
        "quantize_linear_node_0",
        "quantize_linear_node_1",
        "matmul_node_quant",
        "transpose_node",
        "dequantize_linear_node_2",
    ]
    assert [node.name for node in model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
    ]
    assert model.graph.input[0].name == "input_0"
    assert model.graph.output[0].name == "output"


def _test_reshape(model):
    assert [node.name for node in model.graph.node] == [
        "quantize_linear_node_0",
        "quantize_linear_node_1",
        "matmul_node_quant",
        "reshape_node",
        "dequantize_linear_node_2",
    ]
    assert [node.name for node in model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "reshape",
    ]
    assert model.graph.input[0].name == "input_0"
    assert model.graph.output[0].name == "output"


def _test_reshape_and_transpose(model):
    assert [node.name for node in model.graph.node] == [
        "quantize_linear_node_0",
        "quantize_linear_node_1",
        "matmul_node_quant",
        "transpose_node",
        "reshape_node",
        "dequantize_linear_node_2",
    ]
    assert [node.name for node in model.graph.initializer] == [
        "x_scale",
        "y_scale",
        "zero_point",
        "reshape",
    ]
    assert model.graph.input[0].name == "input_0"
    assert model.graph.output[0].name == "output"


@pytest.mark.parametrize(
    "with_transpose, with_reshape, testing_function",
    [
        (False, False, _test_matmul),
        (True, False, _test_transpose),
        (False, True, _test_reshape),
        (True, True, _test_reshape_and_transpose),
    ],
)
def test_convert_quantizable_matmul(with_transpose, with_reshape, testing_function):
    model = _create_test_model(with_reshape=with_reshape, with_transpose=with_transpose)
    transform = MatMulToQLinearMatMul()
    model = transform(model)
    testing_function(model)
    onnx.checker.check_model(model)


def test_convert_quantizable_matmul_without_matmul():
    model_in = _create_model_no_matmul()
    nodes_in = [node.name for node in model_in.graph.node]
    transform = MatMulToQLinearMatMul()
    model_out = transform(model_in)
    onnx.checker.check_model(model_out)
    assert [node.name for node in model_out.graph.node] == nodes_in
