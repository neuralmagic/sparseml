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
from onnx import ModelProto

from sparseml.exporters.transforms.fold_conv_div_bn import FoldConvDivBn
from sparsezoo.utils import validate_onnx


@pytest.fixture
def onnx_model():
    """See docstring of FoldConvDivBn"""
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    weight = onnx.helper.make_tensor(
        name="weight", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
    )
    scale = onnx.helper.make_tensor(
        name="scale", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
    )
    bias = onnx.helper.make_tensor(
        name="bias", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[0]
    )
    mean = onnx.helper.make_tensor(
        name="mean", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[0]
    )
    variance = onnx.helper.make_tensor(
        name="variance", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
    )
    conv = onnx.helper.make_node(
        "Conv", inputs=["input", "weight"], outputs=["conv_output"], name="conv"
    )
    div = onnx.helper.make_node(
        "Div", inputs=["conv_output", "mean"], outputs=["div_output"], name="div"
    )
    bn = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["div_output", "scale", "bias", "mean", "variance"],
        outputs=["output"],
        name="bn",
    )
    graph = onnx.helper.make_graph(
        nodes=[conv, div, bn],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[weight, scale, bias, mean, variance],
    )

    model = onnx.helper.make_model(graph)
    validate_onnx(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = FoldConvDivBn().apply(onnx_model)
    validate_onnx(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["conv"]
    assert [i.name for i in onnx_model.graph.initializer] == ["weight", "conv.bias"]
    assert onnx_model.graph.node[0].op_type == "Conv"


def test_overwrites_existing_bias(onnx_model: ModelProto):
    onnx_model.graph.initializer.append(
        onnx.helper.make_tensor(
            name="test_bias", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
        )
    )
    onnx_model.graph.node[0].input.append("test_bias")
    validate_onnx(onnx_model)

    onnx_model = FoldConvDivBn().apply(onnx_model)
    validate_onnx(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["conv"]
    assert [i.name for i in onnx_model.graph.initializer] == ["weight", "conv.bias"]
