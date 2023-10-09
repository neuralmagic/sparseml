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

from sparseml.core.recipe import Recipe


def _valid_recipes():
    return [
        """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
        """,
        """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
                MagnitudePruningModifier:
                    start: 5
                    end: 10
        """,
        """
        test1_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
        test2_stage:
                MagnitudePruningModifier:
                    start: 5
                    end: 10
        """,
        """
        test1_stage:
            constant_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
            magnitude_modifiers:
                MagnitudePruningModifier:
                    start: 5
                    end: 10
        """,
    ]


@pytest.mark.parametrize("recipe_str", _valid_recipes())
def test_recipe_create_instance_accepts_valid_recipe_string(recipe_str):
    recipe = Recipe.create_instance(recipe_str)
    assert recipe is not None, "Recipe could not be created from string"


@pytest.mark.parametrize("recipe_str", _valid_recipes())
def test_recipe_create_instance_accepts_valid_recipe_file(recipe_str):
    content = yaml.safe_load(recipe_str)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(content, f)
        recipe = Recipe.create_instance(f.name)
        assert recipe is not None, "Recipe could not be created from file"


@pytest.mark.parametrize("recipe_str", _valid_recipes())
def test_serialization(recipe_str):
    recipe_instance = Recipe.create_instance(recipe_str)
    serialized_recipe_str = recipe_instance.yaml()
    recipe_from_serialized = Recipe.create_instance(serialized_recipe_str)

    expected_dict = recipe_instance.dict()
    actual_dict = recipe_from_serialized.dict()

    assert expected_dict == actual_dict
