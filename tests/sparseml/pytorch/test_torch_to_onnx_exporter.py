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
import onnxruntime as ort
import pytest
import torch

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.sparsification.quantization import QuantizationModifier
from sparseml.pytorch.torch_to_onnx_exporter import TorchToONNX
from sparseml.pytorch.utils import ModuleExporter
from tests.sparseml.pytorch.helpers import ConvNet, LinearNet, MLPNet


@pytest.mark.parametrize(
    "model,sample_batch",
    [
        (MLPNet(), torch.randn(8)),
        (MLPNet(), torch.randn(10, 8)),
        (LinearNet(), torch.randn(8)),
        (LinearNet(), torch.randn(10, 8)),
        (ConvNet(), torch.randn(1, 3, 28, 28)),
    ],
)
def test_simple_models_against_module_exporter(tmp_path, model, sample_batch):
    old_exporter = ModuleExporter(model, tmp_path / "old_exporter")
    old_exporter.export_onnx(sample_batch, convert_qat=False)

    (tmp_path / "new_exporter").mkdir()
    new_exporter = TorchToONNX(sample_batch)
    new_exporter.export(model, tmp_path / "new_exporter" / "model.onnx")

    _assert_onnx_models_are_equal(
        str(tmp_path / "old_exporter" / "model.onnx"),
        str(tmp_path / "new_exporter" / "model.onnx"),
        sample_batch,
    )


@pytest.mark.parametrize("quantize", [False, True])
def test_resnet50_exporters_are_equivalent(tmp_path, quantize: bool):
    old_dir = tmp_path / "old_exporter"
    old_dir.mkdir()
    new_dir = tmp_path / "new_exporter"
    new_dir.mkdir()

    sample_batch = torch.rand(10, 3, 224, 224)
    model = ModelRegistry.create(
        "resnet50",
        pretrained=True,
        pretrained_path="zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
    )
    if quantize:
        QuantizationModifier().apply(model)

    old_exporter = ModuleExporter(model, old_dir)
    old_exporter.export_onnx(sample_batch, convert_qat=False)

    new_exporter = TorchToONNX(sample_batch)
    new_exporter.export(model, new_dir / "model.onnx")
    _assert_onnx_models_are_equal(
        str(tmp_path / "old_exporter" / "model.onnx"),
        str(tmp_path / "new_exporter" / "model.onnx"),
        sample_batch,
        expect_op_types=["QuantizeLinear", "DequantizeLinear"] if quantize else None,
    )


@pytest.mark.parametrize("quantize", [False, True])
def test_mobilenet_exporters_are_equivalent(quantize: bool):
    assert False, "TODO"


@pytest.mark.parametrize("quantize", [False, True])
def test_yolov5_exporters_are_equivalent(quantize: bool):
    assert False, "TODO"


@pytest.mark.parametrize("quantize", [False, True])
def test_bert_exporters_are_equivalent(quantize: bool):
    assert False, "TODO"


def _assert_onnx_models_are_equal(
    old_model_path: str,
    new_model_path: str,
    sample_batch: torch.Tensor,
    expect_op_types=None,
):
    onnx.checker.check_model(old_model_path)
    onnx.checker.check_model(new_model_path)

    old_model = onnx.load(old_model_path)
    new_model = onnx.load(new_model_path)

    assert len(old_model.graph.node) == len(new_model.graph.node)
    assert len(old_model.graph.initializer) == len(new_model.graph.initializer)
    old_op_types = set([n.op_type for n in old_model.graph.node])
    new_op_types = set([n.op_type for n in new_model.graph.node])
    assert old_op_types == new_op_types
    if expect_op_types is not None:
        for op_type in expect_op_types:
            assert op_type in old_op_types

    old_session = ort.InferenceSession(old_model_path)
    input_name = old_session.get_inputs()[0].name
    output_name = old_session.get_outputs()[0].name
    old_output = old_session.run([output_name], {input_name: sample_batch.numpy()})

    new_session = ort.InferenceSession(new_model_path)
    new_output = new_session.run([output_name], {input_name: sample_batch.numpy()})

    assert numpy.allclose(old_output, new_output)
