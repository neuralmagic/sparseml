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

from sparseml.exporters.transforms import QuantizeQATEmbedding
from sparsezoo.utils import validate_onnx


def _create_test_model(with_qdq=False):
    """
     Creates a test model with a convolution node and quantize/dequantize nodes

    |    INPUT    QuantizeLinear (with constant embedding)
    |      |          |
    |      |     DequantizeLinear
    |       |        |
    |         Gather
    |           |
    |       QuantizeLinear (Optional)
    |           |
    |       DequantizeLinear (Optional)
    |           |
    |         OUTPUT
    """

    input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (3,))
    embedding = onnx.helper.make_tensor("embedding", onnx.TensorProto.FLOAT, (1,), [1])
    x_scale = onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, (1,), [1])
    y_scale = onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = onnx.helper.make_tensor("zero_point", onnx.TensorProto.INT8, (1,), [1])

    model_output = onnx.helper.make_tensor_value_info(
        "gather_output", onnx.TensorProto.FLOAT, (3,)
    )
    quantize_linear_node_0 = onnx.helper.make_node(
        "QuantizeLinear",
        ["embedding", "y_scale", "zero_point"],
        ["quant_linear_0_output"],
        name="quantize_linear_node_0",
    )
    dequantize_linear_node_0 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_0_output", "x_scale", "zero_point"],
        ["dequant_linear_0_output"],
        name="dequantize_linear_node_0",
    )

    gather_node = onnx.helper.make_node(
        "Gather",
        ["dequant_linear_0_output", "input"],
        ["gather_output"],
        name="gather_node",
    )

    graph = onnx.helper.make_graph(
        nodes=[
            quantize_linear_node_0,
            dequantize_linear_node_0,
            gather_node,
        ],
        name="g",
        inputs=[input],
        initializer=[x_scale, y_scale, embedding, zero_point],
        outputs=[model_output],
    )
    if with_qdq:
        graph = _add_qdq_nodes(graph)

    model = onnx.helper.make_model(graph)
    validate_onnx(model)
    return model


def _add_qdq_nodes(graph):
    quantize_linear_node_1 = onnx.helper.make_node(
        "QuantizeLinear",
        ["gather_output", "y_scale", "zero_point"],
        ["quantize_linear_1_output"],
        name="quantize_linear_node_1",
    )
    dequantize_linear_node_1 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quantize_linear_1_output", "x_scale", "zero_point"],
        ["dequantize_linear_1_output"],
        name="dequantize_linear_node_1",
    )

    graph.node.append(quantize_linear_node_1)
    graph.node.append(dequantize_linear_node_1)
    graph.output[0].name = "dequantize_linear_1_output"
    return graph


def _test_qat_embedding(model):
    assert [node.name for node in model.graph.node] == [
        "gather_node",
        "dequantize_linear_node_0",
    ]
    assert [node.name for node in model.graph.initializer] == [
        "x_scale",
        "zero_point",
        "embedding_quant",
    ]
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "gather_output"


def _test_qat_embedding_w_qdq(model):
    assert [node.name for node in model.graph.node] == [
        "gather_node",
        "dequantize_linear_node_1",
    ]
    assert [node.name for node in model.graph.initializer] == [
        "x_scale",
        "zero_point",
        "embedding_quant",
    ]
    assert model.graph.input[0].name
    assert model.graph.output[0].name == "dequantize_linear_1_output"


@pytest.mark.parametrize(
    "with_qdq, testing_function",
    [
        (False, _test_qat_embedding),
        (True, _test_qat_embedding_w_qdq),
    ],
)
def test_quantize_qat_embedding(with_qdq, testing_function):
    model = _create_test_model(with_qdq=with_qdq)
    transform = QuantizeQATEmbedding()
    model = transform(model)
    testing_function(model)
    validate_onnx(model)
