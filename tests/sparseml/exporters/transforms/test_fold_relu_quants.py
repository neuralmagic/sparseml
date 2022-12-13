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

from sparseml.exporters.transforms.fold_relu_quants import FoldReLUQuants


@pytest.fixture
def onnx_model():
    """See docstring of transform"""
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1])
    relu = onnx.helper.make_node("Relu", ["input"], ["relu_output"], name="relu")
    quant = onnx.helper.make_node(
        "QuantizeLinear", ["relu_output", "scale"], ["quant_output"], name="quant"
    )

    graph = onnx.helper.make_graph(
        nodes=[relu, quant],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[scale],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def onnx_model_with_nonzero_zero_point(onnx_model):
    zp = onnx.helper.make_tensor("zero_point", onnx.TensorProto.UINT8, (1,), [1])
    onnx_model.graph.initializer.append(zp)
    onnx_model.graph.node[-1].input.append("zero_point")
    onnx.checker.check_model(onnx_model)
    return onnx_model


@pytest.fixture
def onnx_model_with_zero_zero_point(onnx_model):
    zp = onnx.helper.make_tensor("zero_point", onnx.TensorProto.UINT8, (1,), [0])
    onnx_model.graph.initializer.append(zp)
    onnx_model.graph.node[-1].input.append("zero_point")
    onnx.checker.check_model(onnx_model)
    return onnx_model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = FoldReLUQuants().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["quant"]
    assert [i.name for i in onnx_model.graph.initializer] == ["scale"]


def test_zero_zp(onnx_model_with_zero_zero_point: ModelProto):
    onnx_model = FoldReLUQuants().apply(onnx_model_with_zero_zero_point)
    onnx.checker.check_model(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["quant"]
    assert [i.name for i in onnx_model.graph.initializer] == ["scale", "zero_point"]


def test_nonzero_zp_does_nothing(onnx_model_with_nonzero_zero_point: ModelProto):
    onnx_model = FoldReLUQuants().apply(onnx_model_with_nonzero_zero_point)
    onnx.checker.check_model(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["relu", "quant"]
    assert [i.name for i in onnx_model.graph.initializer] == ["scale", "zero_point"]
