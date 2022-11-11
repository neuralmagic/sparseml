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

import math
from pathlib import Path

import onnx
import pytest
import torch

from click.testing import CliRunner
from sparseml.pytorch.image_classification.export import main
from sparseml.pytorch.models import resnet18
from sparsezoo.analysis import ModelAnalysis


@pytest.fixture()
def temp_dir(tmp_path):
    """
    return: a temporary directory
    """
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    yield test_dir


@pytest.fixture()
def resnet_checkpoint(temp_dir):
    """
    return: path to a ResNet checkpoint
    """
    checkpoint_path = str(temp_dir / "model.pth")
    model = resnet18()
    torch.save(model, checkpoint_path)
    yield checkpoint_path


@pytest.fixture()
def recipe_path():
    """
    return: path to a resnet recipe
    """
    yield str(Path(__file__).resolve().parent / "resnet_test_recipe.yaml")


def test_export_one_shot(resnet_checkpoint, recipe_path, temp_dir):
    runner = CliRunner()
    arch_key = "resnet18"
    result = runner.invoke(
        main,
        [
            "--arch-key",
            arch_key,
            "--checkpoint-path",
            resnet_checkpoint,
            "--num-samples",
            0,
            "--save-dir",
            temp_dir,
            "--num-classes",
            1000,
            "--one-shot",
            recipe_path,
        ],
    )
    assert result.exit_code == 0

    expected_onnx_path = str(temp_dir / arch_key / "deployment" / "model.onnx")

    # exported file exists
    assert Path(expected_onnx_path).exists()

    # exported file is valid onnx
    model = onnx.load(expected_onnx_path)
    onnx.checker.check_model(model)

    # check onnx model is sparse
    model_analysis = ModelAnalysis.from_onnx(onnx_file_path=expected_onnx_path)

    found_prunable_node: bool = False
    for node in model_analysis.nodes:
        if node.parameterized_prunable:
            found_prunable_node = True
            node_sparsity = node.parameter_summary.block_structure["single"].sparsity
            assert math.isclose(node_sparsity, 0.9, abs_tol=0.01)
    assert found_prunable_node
