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

import os
import tempfile

import numpy
import onnx
import pytest
import torch
from onnx import numpy_helper

from sparseml.pytorch.utils import ModuleExporter
from sparseml.pytorch.utils.exporter import (
    _flatten_qparams,
    _fold_identity_initializers,
)
from tests.sparseml.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_exporter_onnx():
    sample_batch = torch.randn(1, 8)
    exporter = ModuleExporter(MLPNet(), tempfile.gettempdir())
    exporter.export_onnx(sample_batch)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("batch_size", [1, 64])
def test_export_batches(batch_size):
    sample_batch = torch.randn(batch_size, 8)
    exporter = ModuleExporter(MLPNet(), tempfile.gettempdir())
    exporter.export_samples([sample_batch])


def test_fold_identity_initializers():
    mdl_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    mdl_output = onnx.helper.make_tensor_value_info(
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
        inputs=[mdl_input],
        outputs=[mdl_output],
        initializer=[init1],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    assert len(model.graph.node) == 2
    assert len(model.graph.initializer) == 1
    assert [node.name for node in model.graph.node] == ["id1", "add"]

    _fold_identity_initializers(model)

    assert len(model.graph.node) == 1
    assert [node.name for node in model.graph.node] == ["add"]
    assert model.graph.node[0].input == ["init1", "input"]

    onnx.checker.check_model(model)


def test_flatten_params():
    mdl_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    mdl_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )

    zp = onnx.helper.make_tensor(
        name="zero_point", data_type=onnx.TensorProto.UINT8, dims=(1,), vals=[0]
    )
    scale = onnx.helper.make_tensor(
        name="scale", data_type=onnx.TensorProto.FLOAT, dims=(1,), vals=[1.0]
    )
    quantize = onnx.helper.make_node(
        "QuantizeLinear", ["input", "zero_point", "scale"], ["id1_output"], name="id1"
    )

    graph = onnx.helper.make_graph(
        nodes=[quantize],
        name="g",
        inputs=[mdl_input],
        outputs=[mdl_output],
        initializer=[zp, scale],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)

    assert len(model.graph.initializer) == 2
    assert [init.name for init in model.graph.initializer] == ["zero_point", "scale"]
    zp = numpy_helper.to_array(model.graph.initializer[0])
    assert zp.shape == (1,)
    assert zp.dtype == numpy.uint8
    scale = numpy_helper.to_array(model.graph.initializer[1])
    assert scale.shape == (1,)
    assert scale.dtype == numpy.float32

    _flatten_qparams(model)

    assert len(model.graph.initializer) == 2
    assert [init.name for init in model.graph.initializer] == ["zero_point", "scale"]
    zp = numpy_helper.to_array(model.graph.initializer[0])
    assert zp.shape == ()
    assert zp.dtype == numpy.uint8
    scale = numpy_helper.to_array(model.graph.initializer[1])
    assert scale.shape == ()
    assert scale.dtype == numpy.float32

    onnx.checker.check_model(model)
