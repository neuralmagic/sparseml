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

import tempfile

import pytest
import yaml

from sparseml.core.framework import Framework
from sparseml.core.recipe import Recipe
from tests.sparseml.helpers import should_skip_pytorch_tests, valid_recipe_strings


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_recipe_create_instance_accepts_valid_recipe_string(recipe_str):
    recipe = Recipe.create_instance(recipe_str)
    assert recipe is not None, "Recipe could not be created from string"


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_recipe_create_instance_accepts_valid_recipe_file(recipe_str):
    content = yaml.safe_load(recipe_str)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(content, f)
        recipe = Recipe.create_instance(f.name)
        assert recipe is not None, "Recipe could not be created from file"


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_serialization(recipe_str):
    recipe_instance = Recipe.create_instance(recipe_str)
    recipe_from_serialized = Recipe.create_instance(recipe_instance.yaml())

    expected_dict = recipe_instance.dict()
    actual_dict = recipe_from_serialized.dict()

    assert expected_dict == actual_dict


@pytest.mark.skipif(
    should_skip_pytorch_tests(),
    reason="Skipping pytorch tests either torch is not installed or "
    "NM_ML_SKIP_PYTORCH_TESTS is set",
)
def test_recipe_creates_correct_modifier():
    start = 1
    end = 10
    targets = "__ALL_PRUNABLE__"

    yaml_str = f"""
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: {start}
                    end: {end}
                    targets: {targets}
        """

    recipe_instance = Recipe.create_instance(yaml_str)

    stage_modifiers = recipe_instance.create_modifier(framework=Framework.pytorch)
    assert len(stage_modifiers) == 1
    assert len(modifiers:=stage_modifiers[0].modifiers) == 1
    from sparseml.modifiers.pruning.constant.pytorch import (
        ConstantPruningModifierPyTorch,
    )

    assert isinstance(modifier := modifiers[0], ConstantPruningModifierPyTorch)
    assert modifier.start == start
    assert modifier.end == end
