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
from torch.quantization import FakeQuantize

from sparseml.optim.helpers import load_global_recipe_variables_from_yaml
from sparseml.pytorch import recipe_template
from sparseml.pytorch.models import resnet50
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import tensor_sparsity


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
    actual = recipe_template(pruning=pruning, quantization=quantization, **kwargs)
    assert actual
    manager = ScheduledModifierManager.from_yaml(file_path=actual)
    manager.apply(module=model)


@pytest.mark.parametrize(
    "pruning, quantization, kwargs",
    [
        ("true", True, {"lr": "cosine"}),
        ("true", False, {"lr": "cyclic_linear"}),
        ("false", True, {}),
        ("false", False, {}),
    ],
)
def test_recipe_creation_with_a_specific_model(
    pruning: str, quantization: bool, kwargs, model: Optional[Module]
):
    actual = recipe_template(
        pruning=pruning, quantization=quantization, model=model, **kwargs
    )
    assert actual
    manager = ScheduledModifierManager.from_yaml(file_path=actual)
    manager.apply(module=model)


def test_recipe_can_be_updated():
    actual = recipe_template(pruning="true", quantization=False)
    manager = ScheduledModifierManager.from_yaml(
        file_path=actual,
        recipe_variables=dict(
            num_epochs=100,
            lr_func="cosine",
        ),
    )
    recipe_from_manager = str(manager)
    assert "end_epoch: 100" in recipe_from_manager
    assert "lr_func: cosine" in recipe_from_manager


@pytest.mark.parametrize(
    "pruning_algo, expected",
    [
        ("true", "!GlobalMagnitudePruningModifier"),
        ("acdc", "!ACDCPruningModifier"),
        ("mfac", "!MFACPruningModifier"),
        ("constant", "!ConstantPruningModifier"),
    ],
)
def test_pruning_modifiers_match_pruning_algo(pruning_algo: str, expected: str):
    actual_recipe = recipe_template(pruning=pruning_algo)
    manager = ScheduledModifierManager.from_yaml(file_path=actual_recipe)
    manager_recipe = str(manager)
    assert expected in manager_recipe


@pytest.mark.parametrize(
    "pruning_algo, expected",
    [
        ("true", "!MagnitudePruningModifier"),
        ("movement", "!MovementPruningModifier"),
    ],
)
def test_pruning_modifiers_match_pruning_algo_without_global_sparsity(
    pruning_algo: str, expected: str
):
    actual_recipe = recipe_template(pruning=pruning_algo, global_sparsity=False)
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
    actual = recipe_template(pruning=pruning, quantization=quantization, model=model)
    manager = ScheduledModifierManager.from_yaml(file_path=actual)
    manager.apply(module=model)
    model_is_quantized = (
        hasattr(model, "qconfig")
        and model.qconfig is not None
        and any(isinstance(module, FakeQuantize) for module in model.modules())
    )

    if quant_expected:
        assert model_is_quantized
    else:
        assert model_is_quantized is False

    if pruning != "false":
        weights = [weights for _, weights in model.state_dict().items()]
        sparsity = sum(
            tensor_sparsity(weight) * torch.numel(weight) for weight in weights
        ) / sum(torch.numel(weight) for weight in weights)

        assert sparsity > 0.75


@pytest.mark.parametrize(
    "num_epochs, init_lr, final_lr, sparsity, lr_func, num_qat_epochs, "
    "num_pruning_active_epochs",
    [
        (20, 0.001, 0.0, 0.8, "linear", 5, 7.5),
        (3, 0.0001, 0.0, 0.9, "cyclic_linear", 2, 0.5),
    ],
)
def test_correct_recipe_variables(
    num_epochs,
    init_lr,
    final_lr,
    sparsity,
    lr_func,
    num_qat_epochs,
    num_pruning_active_epochs,
):
    actual = recipe_template(
        pruning="true",
        quantization=True,
        num_epochs=num_epochs,
        init_lr=init_lr,
        final_lr=final_lr,
        sparsity=sparsity,
        lr=lr_func,
    )

    actual_recipe_variables = load_global_recipe_variables_from_yaml(actual)

    expected_variables = {
        "num_qat_epochs": num_qat_epochs,
        "num_pruning_active_epochs": num_pruning_active_epochs,
        "num_pruning_finetuning_epochs": num_pruning_active_epochs,
        "num_qat_finetuning_epochs": num_qat_epochs / 2,
        "init_lr": init_lr,
        "final_lr": final_lr,
        "lr_func": lr_func,
        "pruning_init_sparsity": min(0.05, sparsity),
        "pruning_final_sparsity": sparsity,
        "pruning_update_frequency": (
            1 if num_pruning_active_epochs > 20 else num_pruning_active_epochs / 20.0
        ),
        "global_sparsity": True,
    }

    for key, expected_value in expected_variables.items():
        actual_value = actual_recipe_variables.get(key)
        assert actual_value is not None
        assert actual_value == expected_value


@pytest.mark.parametrize(
    "pruning, quantization, distillation",
    [
        ("true", True, True),
        ("true", False, False),
        ("false", True, True),
        ("false", False, False),
    ],
)
def test_distillation(pruning, quantization, distillation):
    actual_recipe = recipe_template(
        pruning=pruning, quantization=quantization, distillation=distillation
    )
    manager = ScheduledModifierManager.from_yaml(file_path=actual_recipe)
    manager_recipe = str(manager)
    recipe_contains_distillation = "!DistillationModifier" in manager_recipe
    if distillation:
        assert recipe_contains_distillation
    else:
        assert recipe_contains_distillation is False


@pytest.mark.parametrize(
    "hardness, temperature, distillation",
    [
        (0.5, 1.0, True),
        (0.5, 1.0, False),
    ],
)
def test_distillation_recipe_variables_can_be_overridden(
    hardness, temperature, distillation
):
    recipe = recipe_template(
        distillation=distillation, hardness=hardness, temperature=temperature
    )

    if distillation:
        assert f"distillation_hardness: {hardness}" in recipe
        assert f"distillation_temperature: {temperature}" in recipe
    else:
        assert "distillation_hardness:" not in recipe
        assert "distillation_temperature:" not in recipe
