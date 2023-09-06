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
from typing import Any, Dict, List, Tuple, Union

import yaml
from pydantic import Field, root_validator

from sparseml.core.framework import Framework
from sparseml.core.modifier import StageModifiers
from sparseml.core.recipe.args import RecipeArgs
from sparseml.core.recipe.base import RecipeBase
from sparseml.core.recipe.metadata import RecipeMetaData
from sparseml.core.recipe.stage import RecipeStage


__all__ = ["Recipe"]


class Recipe(RecipeBase):
    @staticmethod
    def create_instance(path: str) -> "Recipe":
        if not os.path.isfile(path):
            # not a local file, load from SparseZoo
            raise NotImplementedError()

        with open(path, "r") as file:
            content = file.read()

            if path.lower().endswith(".json"):
                obj = json.loads(content)
            elif path.lower().endswith(".yaml") or path.lower().endswith(".yml"):
                obj = yaml.safe_load(content)
            else:
                try:
                    obj = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        obj = yaml.safe_load(content)
                    except yaml.YAMLError:
                        raise ValueError(f"Could not parse recipe from path {path}")

            return Recipe.parse_obj(obj)

    @staticmethod
    def simplify_recipe(
        recipe: "Recipe", stages: List[str], args: Dict[str, Any], shift: int = None
    ) -> "Recipe":
        simplified = Recipe()
        simplified.version = recipe.version
        simplified.args = recipe.args
        simplified.stages = [
            stage
            for stage in recipe.stages
            if ((not stages or "default" in stages) and not stage.exclude_default)
            or stage.group in stages
        ]
        simplified.evaluate(args=args, shift=shift)

        return simplified

    @staticmethod
    def simplify_combine_recipes(
        recipes: List[Union["Recipe", Tuple["Recipe", str, Dict[str, Any]]]]
    ) -> "Recipe":
        simplified = Recipe()

        for recipe_tuple in recipes:
            recipe = (
                recipe_tuple[0] if isinstance(recipe_tuple, tuple) else recipe_tuple
            )
            stages = (
                recipe_tuple[1].split(",") if isinstance(recipe_tuple, tuple) else None
            )
            args = recipe_tuple[2] if isinstance(recipe_tuple, tuple) else None
            recipe_simple = Recipe.simplify_recipe(
                recipe=recipe,
                stages=stages,
                args=args,
                shift=simplified.calculate_end(),
            )
            simplified.version = recipe_simple.version
            simplified.stages.extend(recipe_simple.stages)

        return simplified

    version: str = None
    args: RecipeArgs = None
    stages: List[RecipeStage] = Field(default_factory=list)
    metadata: RecipeMetaData = None
    _args_evaluated: RecipeArgs = None

    def calculate_start(self) -> int:
        return min(
            stage.calculate_start()
            for stage in self.stages
            if stage.calculate_start() >= 0
        )

    def calculate_end(self) -> int:
        return max(
            stage.calculate_end() for stage in self.stages if stage.calculate_end() >= 0
        )

    def evaluate(self, args: Dict[str, Any] = None, shift: int = None):
        args = self.args.combine(args) if self.args else RecipeArgs(**(args or {}))
        self._args_evaluated = args.evaluate()
        for stage in self.stages:
            stage.evaluate(self._args_evaluated, shift)

    def create_modifier(self, framework: Framework) -> List[StageModifiers]:
        if self._args_evaluated is None:
            self.evaluate()
        modifiers = []

        for index, stage in enumerate(self.stages):
            stage_modifiers = stage.create_modifiers(framework)
            stage_modifiers.index = index
            stage_modifiers.group = stage.group
            modifiers.append(stage_modifiers)

        return modifiers

    @root_validator(pre=True)
    def remap_stages(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        modifiers = RecipeStage._combine_modifiers(values)
        stages = [{"modifiers": modifiers, "group": "default"}] if modifiers else []
        add_stages, remove_keys = Recipe._combine_stages(values)
        stages.extend(add_stages)

        for key in remove_keys:
            del values[key]

        values["stages"] = Recipe._combine_stages(values)

        return values

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        dict_ = super().dict(*args, **kwargs)

        for stage in dict_["stages"]:
            name = f"{stage['group']}_stage"
            del stage["group"]
            dict_[name] = stage["args"]

        del dict_["stages"]

        return dict_

    @staticmethod
    def _combine_stages(
        values: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        stages = []
        keys = []

        for key, value in list(values.items()):
            if key.endswith("_stage"):
                keys.append(key)
                value["group"] = key.rsplit("_stage", 1)[0]
                stages.append(value)

        return stages, keys
