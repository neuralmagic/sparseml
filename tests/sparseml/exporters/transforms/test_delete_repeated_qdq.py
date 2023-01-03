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

from sparseml.exporters.transforms.delete_repeated_qdq import DeleteRepeatedQdq


@pytest.fixture
def onnx_model():
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1])
    quant1 = onnx.helper.make_node(
        "QuantizeLinear", ["input", "scale"], ["quant1_output"], name="quant1"
    )
    dequant1 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant1_output", "scale"],
        ["dequant1_output"],
        name="dequant1",
    )
    quant2 = onnx.helper.make_node(
        "QuantizeLinear", ["dequant1_output", "scale"], ["quant2_output"], name="quant2"
    )
    dequant2 = onnx.helper.make_node(
        "DequantizeLinear",
        ["quant2_output", "scale"],
        ["dequant2_output"],
        name="dequant2",
    )

    graph = onnx.helper.make_graph(
        nodes=[quant1, dequant1, quant2, dequant2],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[scale],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = DeleteRepeatedQdq().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["quant2", "dequant2"]
    assert [i.name for i in onnx_model.graph.initializer] == ["scale"]
