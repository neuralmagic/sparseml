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

import numpy
import onnx
import pytest
from onnx import numpy_helper

from sparseml.exporters.transforms.skip_input_quantize import SkipInputQuantize
from sparsezoo.utils import validate_onnx


def onnx_model(input_fp32=False, add_quantize_node=False):
    """
    | Creates an ONNX model:
    |       INPUT
    |         |
    |         |
    |   QuantizeLinear (Optional)
    |         |
    |        ADD (w. constant addend)
    |         |
    |       OUTPUT
    """
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT if input_fp32 else onnx.TensorProto.INT8, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    init1 = numpy_helper.from_array(
        numpy.array([-1.0], dtype=numpy.int8),
        "constant",
    )
    if add_quantize_node:
        scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1])
        zero_point = onnx.helper.make_tensor(
            "zero_point", onnx.TensorProto.INT8, (1,), [1]
        )
        quantize = onnx.helper.make_node(
            "QuantizeLinear",
            ["input", "scale", "zero_point"],
            ["quantize_output"],
            name="quantize",
        )
        add = onnx.helper.make_node(
            "Add", ["quantize_output", "constant"], ["output"], name="add"
        )
        nodes = [quantize, add]
        initializer = [scale, zero_point, init1]
    else:
        add = onnx.helper.make_node(
            "Add", ["input", "constant"], ["output"], name="add"
        )
        nodes = [add]
        initializer = [init1]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=initializer,
    )

    model = onnx.helper.make_model(graph)
    validate_onnx(model)
    return model


def _test_modified(model):
    assert [n.name for n in model.graph.node] == ["add"]
    assert [i.name for i in model.graph.initializer] == ["constant"]
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"


def _test_not_modified(model):
    assert [n.name for n in model.graph.node] == ["quantize", "add"]
    assert [i.name for i in model.graph.initializer] == [
        "scale",
        "zero_point",
        "constant",
    ]
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"


@pytest.mark.parametrize(
    "input_fp32, add_quantize_node, testing_func",
    [
        (False, False, _test_modified),
        (False, True, _test_not_modified),
        (True, False, _test_modified),
        (True, True, _test_modified),
    ],
)
def test_skip_input_quantize(input_fp32, add_quantize_node, testing_func):
    model = onnx_model(input_fp32, add_quantize_node)
    transform = SkipInputQuantize()
    model = transform.apply(model)
    validate_onnx(model)
    testing_func(model)
