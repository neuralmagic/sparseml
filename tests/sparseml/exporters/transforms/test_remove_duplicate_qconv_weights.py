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
from onnx import ModelProto, numpy_helper

from sparseml.exporters.transforms.remove_duplicate_qconv_weights import (
    RemoveDuplicateQConvWeights,
)
from sparsezoo.utils import validate_onnx


@pytest.fixture
def onnx_model():
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output_1 = onnx.helper.make_tensor_value_info(
        "add1_output", onnx.TensorProto.FLOAT, (1,)
    )
    model_output_2 = onnx.helper.make_tensor_value_info(
        "add2_output", onnx.TensorProto.FLOAT, (1,)
    )
    model_output_3 = onnx.helper.make_tensor_value_info(
        "add3_output", onnx.TensorProto.FLOAT, (1,)
    )
    model_output_4 = onnx.helper.make_tensor_value_info(
        "conv4_output", onnx.TensorProto.FLOAT, (1,)
    )
    model_output_5 = onnx.helper.make_tensor_value_info(
        "conv5_output", onnx.TensorProto.FLOAT, (1,)
    )
    model_output_6 = onnx.helper.make_tensor_value_info(
        "conv6_output", onnx.TensorProto.FLOAT, (1,)
    )
    zp = onnx.helper.make_tensor("zp", onnx.TensorProto.UINT8, (1,), [0])
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1.0])
    weight1_a = onnx.helper.make_tensor("weight1_a", onnx.TensorProto.UINT8, (1,), [1])
    weight1_b = onnx.helper.make_tensor("weight1_b", onnx.TensorProto.UINT8, (1,), [1])
    weight2 = onnx.helper.make_tensor("weight2", onnx.TensorProto.UINT8, (1,), [2])
    weight3_a = onnx.helper.make_tensor("weight3_a", onnx.TensorProto.UINT8, (1,), [3])
    weight3_b = onnx.helper.make_tensor("weight3_b", onnx.TensorProto.UINT8, (1,), [3])
    weight3_c = onnx.helper.make_tensor("weight3_c", onnx.TensorProto.UINT8, (1,), [3])

    conv1 = onnx.helper.make_node(
        "ConvInteger", ["input", "weight1_a"], ["conv1_output"], name="conv1"
    )
    conv2 = onnx.helper.make_node(
        "QLinearConv",
        ["input", "scale", "zp", "weight1_b", "scale", "zp", "scale", "zp"],
        ["conv2_output"],
        name="conv2",
    )

    conv3 = onnx.helper.make_node(
        "ConvInteger", ["input", "weight2"], ["conv3_output"], name="conv3"
    )

    conv4 = onnx.helper.make_node(
        "QLinearConv",
        ["input", "scale", "zp", "weight3_a", "scale", "zp", "scale", "zp"],
        ["conv4_output"],
        name="conv4",
    )
    conv5 = onnx.helper.make_node(
        "QLinearConv",
        ["input", "scale", "zp", "weight3_b", "scale", "zp", "scale", "zp"],
        ["conv5_output"],
        name="conv5",
    )
    conv6 = onnx.helper.make_node(
        "ConvInteger", ["input", "weight3_c"], ["conv6_output"], name="conv6"
    )
    add1 = onnx.helper.make_node(
        "Add",
        ["conv1_output", "conv2_output"],
        ["add1_output"],
        name="add1",
    )
    add2 = onnx.helper.make_node(
        "Add",
        ["conv3_output", "conv4_output"],
        ["add2_output"],
        name="add2",
    )
    add3 = onnx.helper.make_node(
        "Add",
        ["conv5_output", "conv6_output"],
        ["add3_output"],
        name="add3",
    )
    graph = onnx.helper.make_graph(
        nodes=[conv1, conv2, conv3, conv4, conv5, conv6, add1, add2, add3],
        name="g",
        inputs=[model_input],
        outputs=[
            model_output_1,
            model_output_2,
            model_output_3,
            model_output_4,
            model_output_5,
            model_output_6,
        ],
        initializer=[
            weight1_a,
            weight1_b,
            weight2,
            weight3_a,
            weight3_b,
            weight3_c,
            scale,
            zp,
        ],
    )

    model = onnx.helper.make_model(graph)
    validate_onnx(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = RemoveDuplicateQConvWeights().apply(onnx_model)
    validate_onnx(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "conv6",
        "add1",
        "add2",
        "add3",
    ]
    assert [i.name for i in onnx_model.graph.initializer] == [
        "weight2",
        "scale",
        "zp",
        "qconv_shared_weight_group_0",
        "qconv_shared_weight_group_2",
    ]

    g1, _, _, g0, g2 = onnx_model.graph.initializer

    assert list(numpy_helper.to_array(g0)) == [1]
    assert list(numpy_helper.to_array(g1)) == [2]
    assert list(numpy_helper.to_array(g2)) == [3]
