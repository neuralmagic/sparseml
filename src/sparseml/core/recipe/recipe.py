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

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, root_validator

from sparseml.core.framework import Framework
from sparseml.core.modifier import StageModifiers
from sparseml.core.recipe.args import RecipeArgs
from sparseml.core.recipe.metadata import RecipeMetaData
from sparseml.core.recipe.stage import RecipeStage


__all__ = ["Recipe"]


class Recipe(BaseModel):
    version: str = None
    args: RecipeArgs = None
    stages: List[RecipeStage] = Field(default_factory=list)
    metadata: RecipeMetaData = None
    _args_evaluated: RecipeArgs = None

    def evaluate(self, args: Dict[str, Any] = None, shift: int = None):
        args = self.args.combine(args)
        self._args_evaluated = args.evaluate()
        for stage in self.stages:
            stage.evaluate(self._args_evaluated, shift)

    def create_modifiers(self, framework: Framework) -> List[StageModifiers]:
        self.evaluate()
        modifiers = []

        for index, stage in enumerate(self.stages):
            stage_modifiers = stage.create_modifiers(framework)
            stage_modifiers.index = index
            stage_modifiers.group = stage.group
            modifiers.append(stage_modifiers)

        return stage_modifiers

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
