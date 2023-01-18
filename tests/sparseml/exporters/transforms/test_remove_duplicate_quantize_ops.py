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

from sparseml.exporters.transforms.remove_duplicate_quantize_ops import (
    RemoveDuplicateQuantizeOps,
)


@pytest.fixture
def onnx_model():
    """
    input
    |     |
    quant1  quant2
    """
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    zp = onnx.helper.make_tensor("zp", onnx.TensorProto.UINT8, (1,), [0])
    scale1 = onnx.helper.make_tensor("scale1", onnx.TensorProto.FLOAT, (1,), [1.0])

    # group 1
    quant1 = onnx.helper.make_node(
        "QuantizeLinear", ["input", "scale1", "zp"], ["quant1_output"], name="quant1"
    )
    quant2 = onnx.helper.make_node(
        "QuantizeLinear", ["input", "scale1", "zp"], ["quant2_output"], name="quant2"
    )

    graph = onnx.helper.make_graph(
        nodes=[quant1, quant2],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[scale1, zp],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = RemoveDuplicateQuantizeOps().apply(onnx_model)
    onnx.checker.check_model(onnx_model)
    assert [n.name for n in onnx_model.graph.node] == ["quant1"]
