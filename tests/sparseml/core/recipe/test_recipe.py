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

import yaml

from sparseml.core.recipe import Recipe


def _valid_recipe():
    return """
        test_stage:
            pruning_modifiers:
                ConstantPruningModifier:
                    start: 0
                    end: 5
        """


def test_recipe_create_instance_accepts_valid_recipe_string():
    test_recipe = _valid_recipe()
    recipe = Recipe.create_instance(test_recipe)
    assert recipe is not None, "Recipe could not be created from string"


def test_recipe_create_instance_accepts_valid_recipe_file():
    content = yaml.safe_load(_valid_recipe())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(content, f)
        recipe = Recipe.create_instance(f.name)
        assert recipe is not None, "Recipe could not be created from file"
