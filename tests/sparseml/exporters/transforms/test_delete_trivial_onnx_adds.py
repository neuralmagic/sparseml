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

from sparseml.exporters.transforms import DeleteTrivialOnnxAdds
from sparsezoo.utils import validate_onnx


def _create_test_model(initializer_set_to_nonzero=False):
    """
    | Creates an ONNX model:
    |   Input   Constant (with initializer, optionally set to zero)
    |       |    |
    |        ADD
    |         |
    |       OUTPUT
    """
    if initializer_set_to_nonzero:
        initializer_vals = [0, 1]
    else:
        initializer_vals = [0, 0]

    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (2,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (2,)
    )
    initializer_constant = onnx.helper.make_tensor(
        name="constant_initializer",
        data_type=onnx.TensorProto.FLOAT,
        dims=(2,),
        vals=initializer_vals,
    )
    constant = onnx.helper.make_node(
        "Constant", [], ["constant_output"], value=initializer_constant, name="constant"
    )
    add = onnx.helper.make_node(
        "Add", ["input", "constant_output"], ["output"], name="add"
    )

    graph = onnx.helper.make_graph(
        nodes=[constant, add],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[],
    )

    model = onnx.helper.make_model(graph)

    return model


def _test_initializer_zero(model):
    assert not [node.name for node in model.graph.node]
    assert not [node.name for node in model.graph.initializer]
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"


def _test_initializer_nonzero(model):
    assert [node.name for node in model.graph.node] == ["constant", "add"]
    assert [node.name for node in model.graph.initializer] == []
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"


@pytest.mark.parametrize(
    "initializer_set_to_nonzero, test_function",
    [(True, _test_initializer_nonzero), (False, _test_initializer_zero)],
)
def test_delete_trivial_onnx_adds(initializer_set_to_nonzero, test_function):
    model = _create_test_model(initializer_set_to_nonzero=initializer_set_to_nonzero)
    validate_onnx(model)
    transform = DeleteTrivialOnnxAdds()
    model = transform(model)
    validate_onnx(model)
    test_function(model)
