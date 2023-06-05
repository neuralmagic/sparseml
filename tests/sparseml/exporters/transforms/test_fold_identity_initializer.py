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

from sparseml.exporters.transforms import FoldIdentityInitializers
from sparsezoo.utils import validate_onnx


def _create_test_model():
    """
    | Creates an ONNX model:
    |   INPUT   Identity (with initializer)
    |       |    |
    |        ADD
    |         |
    |       OUTPUT
    """
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    init1 = onnx.helper.make_tensor(
        name="init1", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1]
    )
    id1 = onnx.helper.make_node("Identity", ["init1"], ["id1_output"], name="id1")
    add = onnx.helper.make_node("Add", ["id1_output", "input"], ["output"], name="add")

    graph = onnx.helper.make_graph(
        nodes=[id1, add],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[init1],
    )

    model = onnx.helper.make_model(graph)

    def _test_created_model(model):
        assert len(model.graph.node) == 2
        assert len(model.graph.initializer) == 1
        assert [node.name for node in model.graph.node] == ["id1", "add"]

    _test_created_model(model)

    return model


def _test_result(model):
    assert [node.name for node in model.graph.node] == ["add"]


def test_fold_identity_initializers():
    model = _create_test_model()
    validate_onnx(model)
    transform = FoldIdentityInitializers()
    model = transform(model)
    validate_onnx(model)
    _test_result(model)
