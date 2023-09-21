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

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import yaml
from pydantic import Field, root_validator

from sparseml.core.framework import Framework
from sparseml.core.recipe.args import RecipeArgs
from sparseml.core.recipe.base import RecipeBase
from sparseml.core.recipe.metadata import RecipeMetaData
from sparseml.core.recipe.stage import RecipeStage


__all__ = ["Recipe", "RecipeTuple"]


class Recipe(RecipeBase):
    @staticmethod
    def create_instance(path: str) -> "Recipe":
        """
        Create a recipe instance from a file, or string

        :param path: The path to the recipe file or
            SparseZoo stub or the recipe string, must be a valid
            json/yaml file or a valid json/yaml string
        """
        if not os.path.isfile(path):
            # not a local file
            if path.startswith("zoo:"):
                # download from SparseZoo
                raise NotImplementedError("Using SparseZoo stubs is not yet supported")
            else:
                # assume it's a string
                obj = _load_json_or_yaml_string(path)
                return Recipe.parse_obj(obj)

        with open(path, "r") as file:
            content = file.read().strip()

            if path.lower().endswith(".json"):
                obj = json.loads(content)
            elif path.lower().endswith(".yaml") or path.lower().endswith(".yml"):
                obj = yaml.safe_load(content)
            else:
                try:
                    obj = _load_json_or_yaml_string(content)
                except ValueError:
                    raise ValueError(f"Could not parse recipe from path {path}")
            return Recipe.parse_obj(obj)

    @staticmethod
    def simplify_recipe(
        recipe: Union["Recipe", "RecipeTuple"], shift: int = None
    ) -> "Recipe":
        stages = []
        if isinstance(recipe, RecipeTuple):
            stage_names = recipe.target_stages
            if stage_names is None:
                stages = recipe.recipe.stages
            else:
                for stage in recipe.recipe.stages:
                    if stage.group in stage_names:
                        stages.append(stage)
        args = recipe.override_args if isinstance(recipe, RecipeTuple) else {}
        version = recipe.version if isinstance(recipe, Recipe) else None

        simplified = Recipe()
        simplified.version = version
        simplified.args = RecipeArgs(args)
        simplified.stages = stages
        simplified.evaluate(args=args, shift=shift)

        return simplified

    @staticmethod
    def simplify_combine_recipes(
        recipes: List[Union["Recipe", "RecipeTuple"]]
    ) -> "Recipe":

        combined = Recipe()

        for recipe in recipes:
            simplified = Recipe.simplify_recipe(
                recipe=recipe,
                shift=combined.calculate_end(),
            )
            combined.version = simplified.version
            combined.stages.extend(simplified.stages)
            combined.args.combine(simplified.args)

        return combined

    version: str = None
    args: RecipeArgs = Field(default_factory=RecipeArgs)
    stages: List[RecipeStage] = Field(default_factory=list)
    metadata: RecipeMetaData = None
    args_evaluated: RecipeArgs = Field(default_factory=RecipeArgs)

    def calculate_start(self) -> int:
        return min(
            stage.calculate_start()
            for stage in self.stages
            if stage.calculate_start() >= 0
        )

    def calculate_end(self) -> int:
        if len(self.stages) == 0:
            return 0
        return max(
            stage.calculate_end() for stage in self.stages if stage.calculate_end() >= 0
        )

    def evaluate(self, args: Dict[str, Any] = None, shift: int = None):
        args = self.args.combine(args) if self.args else RecipeArgs(**(args or {}))
        self.args_evaluated = args.evaluate()
        for stage in self.stages:
            stage.evaluate(self.args_evaluated, shift)

    def create_modifier(self, framework: Framework) -> List["StageModifiers"]:
        if self.args_evaluated is None:
            self.evaluate()
        modifiers = []

        for index, stage in enumerate(self.stages):
            stage_modifiers = stage.create_modifier(framework)
            stage_modifiers.index = index
            stage_modifiers.group = stage.group
            modifiers.append(stage_modifiers)

        return modifiers

    @root_validator(pre=True)
    def remap_stages(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        stages = []

        modifiers = RecipeStage.extract_dict_modifiers(values)
        if modifiers:
            default_stage = {"modifiers": modifiers, "group": "default"}
            stages.append(default_stage)

        extracted = Recipe.extract_dict_stages(values)
        stages.extend(extracted)
        values["stages"] = stages

        return values

    @staticmethod
    def extract_dict_stages(values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Accepted stage formats:
        - stages:
          first_stage:
            modifiers: ...
          second_stage:
            modifiers: ...

        - first_stage:
          modifiers: ...
        - second_stage:
          modifiers: ...

        Accepted modifier formats default stage:
        - modifiers:
          - ModifierTypeOne
            ...
          - ModifierTypeTwo
            ...

        - first_modifiers:
          - ModifierTypeOne
            ...
          - ModifierTypeTwo
            ...
        """

        stages = []
        remove_keys = []

        default_modifiers = RecipeStage.extract_dict_modifiers(values)
        if default_modifiers:
            default_stage = {"modifiers": default_modifiers, "group": "default"}
            stages.append(default_stage)

        if "stages" in values and values["stages"]:
            assert isinstance(
                values["stages"], dict
            ), f"stages must be a dict, given {values['stages']}"
            remove_keys.append("stages")

            for key, value in values["stages"].items():
                assert isinstance(value, dict), f"stage must be a dict, given {value}"
                value["group"] = key
                stages.append(value)

        for key, value in list(values.items()):
            if key.endswith("_stage"):
                remove_keys.append(key)
                value["group"] = key.rsplit("_stage", 1)[0]
                stages.append(value)

        for key in remove_keys:
            del values[key]

        return stages

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        dict_ = super().dict(*args, **kwargs)
        stages = {}

        for stage in dict_["stages"]:
            name = stage["group"]
            del stage["group"]

            if name not in stages:
                stages[name] = []

            stages[name].append(stage)

        dict_["stages"] = stages

        return dict_


@dataclass
class RecipeTuple:
    recipe: Recipe
    target_stages: List[str]
    override_args: Dict[str, Any]


def _load_json_or_yaml_string(content: str) -> Dict[str, Any]:
    # try loading as json first, then yaml
    # if both fail, raise a ValueError
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as err:
            raise ValueError(f"Could not parse recipe from string {content}") from err
