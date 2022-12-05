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

from sparseml.exporters.transforms import ConvertQuantizableConvInteger


def _create_test_model():
    """
     Creates a test model with a convolution node and quantize/dequantize nodes

    | Starting with:
     |          INPUT         QuantizeLinear (with constant kernel)
     |            |               |
     |     QuantizeLinear     DequantizeLinear
     |            |               |
     |     DequantizeLinear      |
     |                  |      |
     |                   Conv (with bias)
     |                     |
     |                  OUTPUT
    """
    x_scale = onnx.helper.make_tensor("x_scale", onnx.TensorProto.FLOAT, (1,), [1])
    y_scale = onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = onnx.helper.make_tensor("zero_point", onnx.TensorProto.INT8, (1,), [1])

    bias = onnx.helper.make_tensor("bias", onnx.TensorProto.FLOAT, (3,), [1, 1, 1])

    input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (3, 3))
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (1,))

    quantize_linear_node_0 = onnx.helper.make_node(
        "QuantizeLinear",
        ["input", "y_scale", "zero_point"],
        ["quant_linear_0_output"],
        name="quantize_linear_node_0",
    )
    dequantize_linear_node_0 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_0_output", "x_scale", "zero_point"],
        ["dequant_linear_0_output"],
        name="dequantize_linear_node_0",
    )

    kernel = onnx.helper.make_tensor(
        "kernel", onnx.TensorProto.FLOAT, (3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    quantize_linear_node_1 = onnx.helper.make_node(
        "QuantizeLinear",
        ["kernel", "y_scale", "zero_point"],
        ["quant_linear_1_output"],
        name="quantize_linear_node_1",
    )
    dequantize_linear_node_1 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant_linear_1_output", "x_scale", "zero_point"],
        ["dequant_linear_1_output"],
        name="dequantize_linear_node_1",
    )

    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=["dequant_linear_0_output", "dequant_linear_1_output", "bias"],
        outputs=["conv_node_output"],
        kernel_shape=[3, 3],
        name="conv_node",
    )

    graph = onnx.helper.make_graph(
        nodes=[
            quantize_linear_node_0,
            dequantize_linear_node_0,
            quantize_linear_node_1,
            dequantize_linear_node_1,
            conv_node,
        ],
        name="g",
        inputs=[input],
        outputs=[output],
        initializer=[y_scale, x_scale, bias, zero_point, kernel],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test():
    model = _create_test_model()
    transform = ConvertQuantizableConvInteger()
    model = transform(model)
    onnx.checker.check_model(model)
    assert [node.name for node in model.graph.node] == [
        "quantize_linear_node_0",
        "conv_node_quant",
        "conv_node_bias_add_quant",
        "conv_node_bias_add_quant_cast",
        "conv_node_bias_add_quant_rescale_mul",
    ]
