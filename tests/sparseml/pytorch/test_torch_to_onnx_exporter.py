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
import torch

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
    old_exporter.export_onnx(sample_batch)

    (tmp_path / "new_exporter").mkdir(parents=True)
    new_exporter = TorchToONNX(sample_batch)
    new_exporter.export(model, tmp_path / "new_exporter" / "model.onnx")

    old_model = onnx.load(tmp_path / "old_exporter" / "model.onnx")
    new_model = onnx.load(tmp_path / "new_exporter" / "model.onnx")

    assert len(old_model.graph.node) == len(new_model.graph.node)
    assert len(old_model.graph.initializer) == len(new_model.graph.initializer)


def test_no_pruning_no_quant():
    # TODO mobilenet
    # TODO resnet50
    # TODO yolov5
    # TODO bert
    ...


def test_yes_pruning_no_quant():
    # TODO mobilenet
    # TODO resnet50
    # TODO yolov5
    # TODO bert
    ...


def test_no_pruning_yes_quant():
    # TODO mobilenet
    # TODO resnet50
    # TODO yolov5
    # TODO bert
    ...


def test_yes_pruning_yes_quant():
    # TODO mobilenet
    # TODO resnet50
    # TODO yolov5
    # TODO bert
    ...
