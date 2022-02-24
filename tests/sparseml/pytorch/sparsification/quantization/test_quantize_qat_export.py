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
from onnx import TensorProto

from sparseml.pytorch.sparsification.quantization import skip_onnx_input_quantize


def test_skip_onnx_input_quantize():
    # make sample graph of fp32 input -> QuantizeLinear -> QLinearConv
    # verify that it is transformed to uint8 input -> QLinearConv

    float_input = onnx.helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, None, None]
    )
    quant_node = onnx.helper.make_node(
        "QuantizeLinear",
        ["input", "scale", "zp"],
        ["quant_output"],
    )
    qconv_node = onnx.helper.make_node(
        "QLinearConv",
        ["quant_output", "scale", "zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"],
        ["qconv_output"],
    )

    qconv_output = onnx.helper.make_tensor_value_info(
        "qconv_output", TensorProto.UINT8, [1, 1, None, None]
    )

    graph = onnx.helper.make_graph(
        [quant_node, qconv_node],
        "test_graph",
        [float_input],
        [qconv_output],
        [],
    )
    model = onnx.helper.make_model(graph)

    # initial model checks
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert len(model.graph.node) == 2
    assert model.graph.node[0].op_type == "QuantizeLinear"
    assert model.graph.node[1].op_type == "QLinearConv"

    assert model.graph.node[0].input[0] == model.graph.input[0].name
    assert model.graph.node[1].input[0] == model.graph.node[0].output[0]

    # run optimization
    skip_onnx_input_quantize(model)

    # check model has uint8 inputs and no qlinear input node
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.UINT8
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "QLinearConv"

    assert model.graph.node[0].input[0] == model.graph.input[0].name


def test_skip_onnx_input_quantize_expected_exception():
    # test that a graph with already quantized inputs fails for this optimization

    int_input = onnx.helper.make_tensor_value_info(
        "input", TensorProto.UINT8, [1, 3, None, None]
    )
    qconv_node = onnx.helper.make_node(
        "QLinearConv",
        ["input", "scale", "zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"],
        ["qconv_output"],
    )

    qconv_output = onnx.helper.make_tensor_value_info(
        "qconv_output", TensorProto.UINT8, [1, 1, None, None]
    )

    graph = onnx.helper.make_graph(
        [qconv_node],
        "test_graph",
        [int_input],
        [qconv_output],
        [],
    )
    model = onnx.helper.make_model(graph)
    with pytest.raises(RuntimeError):
        skip_onnx_input_quantize(model)
