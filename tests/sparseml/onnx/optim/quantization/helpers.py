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

"""
Helper functions to test quantization package
"""

import tempfile

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, numpy_helper


__all__ = [
    "make_tmp_onnx_file",
    "onnx_conv_net",
    "onnx_linear_net",
]


def _random_float_tensor(name, *shape):
    return numpy_helper.from_array(
        np.random.rand(*shape).astype(np.float32),
        name,
    )


def make_tmp_onnx_file(model: ModelProto) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    onnx.save(model, path)
    return path


def onnx_conv_net() -> ModelProto:
    """
    :return: ONNX model of input -> conv -> bn -> output
    """
    # Model input
    mdl_input = onnx.helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 5, 5]
    )

    # Conv Layer
    conv1_weight = _random_float_tensor("conv1.weight", 1, 3, 3, 3)
    onnx.helper.make_tensor_value_info("conv1.output", TensorProto.FLOAT, [1, 1, 5, 5])
    conv1_node = onnx.helper.make_node(
        "Conv",
        ["input", "conv1.weight"],
        ["conv1.output"],
        name="Conv1",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # Batch Norm Layer 1
    bn1_scale = _random_float_tensor("bn1.weight", 1)
    bn1_bias = _random_float_tensor("bn1.bias", 1)
    bn1_mean = _random_float_tensor("bn1.running_mean", 1)
    bn1_var = _random_float_tensor("bn1.running_var", 1)
    onnx.helper.make_tensor_value_info("bn1.output", TensorProto.FLOAT, [1, 1, 5, 5])
    bn1_node = onnx.helper.make_node(
        "BatchNormalization",
        [
            "conv1.output",
            "bn1.weight",
            "bn1.bias",
            "bn1.running_mean",
            "bn1.running_var",
        ],
        ["bn1.output"],
        name="BatchNorm1",
    )

    # Conv Layer 2
    conv2_weight = _random_float_tensor("conv2.weight", 1, 1, 3, 3)
    onnx.helper.make_tensor_value_info("conv2.output", TensorProto.FLOAT, [1, 1, 5, 5])
    conv2_node = onnx.helper.make_node(
        "Conv",
        ["bn1.output", "conv2.weight"],
        ["conv2.output"],
        name="Conv2",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # Batch Norm Layer
    bn2_scale = _random_float_tensor("bn2.weight", 1)
    bn2_bias = _random_float_tensor("bn2.bias", 1)
    bn2_mean = _random_float_tensor("bn2.running_mean", 1)
    bn2_var = _random_float_tensor("bn2.running_var", 1)
    bn2_output = onnx.helper.make_tensor_value_info(
        "bn2.output", TensorProto.FLOAT, [1, 1, 5, 5]
    )
    bn2_node = onnx.helper.make_node(
        "BatchNormalization",
        [
            "conv2.output",
            "bn2.weight",
            "bn2.bias",
            "bn2.running_mean",
            "bn2.running_var",
        ],
        ["bn2.output"],
        name="BatchNorm2",
    )

    graph = onnx.helper.make_graph(
        [conv1_node, bn1_node, conv2_node, bn2_node],
        "test_graph",
        [mdl_input],
        [bn2_output],
        [
            conv1_weight,
            bn1_scale,
            bn1_bias,
            bn1_mean,
            bn1_var,
            conv2_weight,
            bn2_scale,
            bn2_bias,
            bn2_mean,
            bn2_var,
        ],
    )
    opset_id = onnx.helper.make_opsetid("", 11)
    model = onnx.helper.make_model(graph, opset_imports=[opset_id], ir_version=6)
    return model


def onnx_linear_net() -> ModelProto:
    """
    :return: ONNX model of input -> matmul -> -> matmul -> output
    """
    # Model input
    mdl_input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [20, 20])

    # MatMul Layer 1
    matmul1_weight = _random_float_tensor("matmul1.weight", 20, 20)
    onnx.helper.make_tensor_value_info("matmul1.output", TensorProto.FLOAT, [20, 20])
    matmul1_node = onnx.helper.make_node(
        "MatMul", ["input", "matmul1.weight"], ["matmul1.output"], name="MatMul1"
    )

    # MatMul Layer 2
    matmul2_weight = _random_float_tensor("matmul2.weight", 20, 1)
    matmul2_output = onnx.helper.make_tensor_value_info(
        "matmul2.output", TensorProto.FLOAT, [20, 1]
    )
    matmul2_node = onnx.helper.make_node(
        "MatMul",
        ["matmul1.output", "matmul2.weight"],
        ["matmul2.output"],
        name="MatMul2",
    )

    # Generate Model
    graph = onnx.helper.make_graph(
        [matmul1_node, matmul2_node],
        "test_graph",
        [mdl_input],
        [matmul2_output],
        [matmul1_weight, matmul2_weight],
    )
    opset_id = onnx.helper.make_opsetid("", 11)
    model = onnx.helper.make_model(graph, opset_imports=[opset_id], ir_version=6)
    return model
