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
from typing import Optional

import pytest
import torch
from packaging import version
from torch.nn import Module

from sparseml.pytorch.models import resnet50
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.recipe_template.main import _build_recipe


@pytest.fixture
def model():
    """
    A generic resnet model to test recipes
    """
    yield resnet50()


min_torch_version = pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.9"),
    reason="requires torch 1.9 or higher",
)


@pytest.mark.parametrize(
    "pruning, quantization, kwargs",
    [
        ("true", True, {}),
        ("true", True, {"global_sparsity": True}),
        ("true", False, {}),
        ("false", True, {}),
        ("false", False, {}),
    ],
)
def test_generic_recipe_creation(
    pruning: str, quantization: bool, kwargs, model: Optional[Module]
):
    actual = _build_recipe(pruning=pruning, quantization=quantization, **kwargs)
    assert actual
    manager = ScheduledModifierManager.from_yaml(file_path=actual)
    manager.apply(module=model)


@pytest.mark.parametrize(
    "pruning, quantization, kwargs",
    [
        ("true", True, {"lr_func": "cosine"}),
        ("true", False, {"lr_func": "cyclic_linear"}),
        ("false", True, {}),
        ("false", False, {}),
    ],
)
def test_recipe_creation_with_a_specific_model(
    pruning: str, quantization: bool, kwargs, model: Optional[Module]
):
    actual = _build_recipe(
        pruning=pruning, quantization=quantization, model=model, **kwargs
    )
    assert actual
    manager = ScheduledModifierManager.from_yaml(file_path=actual)
    manager.apply(module=model)


def test_recipe_can_be_updated():
    actual = _build_recipe(pruning="true", quantization=False)
    manager = ScheduledModifierManager.from_yaml(
        file_path=actual,
        recipe_variables=dict(
            start_epoch=100,
            end_epoch=1000,
        ),
    )
    recipe_from_manager = str(manager)
    assert "start_epoch: 100" in recipe_from_manager
    assert "end_epoch: 1000" in recipe_from_manager


@pytest.mark.parametrize(
    "pruning_algo, expected",
    [
        ("true", "!MagnitudePruningModifier"),
        ("acdc", "!ACDCPruningModifier"),
        ("mfac", "!MFACPruningModifier"),
        ("movement", "!MovementPruningModifier"),
        ("constant", "!ConstantPruningModifier"),
    ],
)
def test_pruning_modifiers_match_pruning_algo(pruning_algo: str, expected: str):
    actual_recipe = _build_recipe(pruning=pruning_algo)
    manager = ScheduledModifierManager.from_yaml(file_path=actual_recipe)
    manager_recipe = str(manager)
    assert expected in manager_recipe


@min_torch_version
def test_obs_modifier():
    test_pruning_modifiers_match_pruning_algo(
        pruning_algo="obs", expected="!OBSPruningModifier"
    )


@pytest.mark.parametrize(
    "pruning, quantization, quant_expected",
    [
        ("true", True, True),
        ("true", False, False),
        ("false", True, True),
        ("false", False, False),
    ],
)
def test_one_shot_applies_sparsification(pruning, quantization, quant_expected, model):
    actual = _build_recipe(pruning=pruning, quantization=quantization, model=model)
    manager = ScheduledModifierManager.from_yaml(file_path=actual)
    manager.apply(module=model)
    model_is_quantized = hasattr(model, "qconfig") and model.qconfig is not None

    if quant_expected:
        assert model_is_quantized
    else:
        assert model_is_quantized is False
