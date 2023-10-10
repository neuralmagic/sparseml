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

from typing import Any, Dict, List, Optional

from pydantic import Field, root_validator

from sparseml.core.framework import Framework
from sparseml.core.modifier import StageModifiers
from sparseml.core.recipe.args import RecipeArgs
from sparseml.core.recipe.base import RecipeBase
from sparseml.core.recipe.modifier import RecipeModifier


__all__ = ["RecipeStage"]


class RecipeStage(RecipeBase):
    """
    Represents a stage in a recipe.

    :param group: Name of the current stage
    :param args: Optional RecipeArgs to use for this stage
    :param enabled: True to enable the stage, False otherwise
    :param modifiers: list of RecipeModifiers that are a part of this stage
    :param exclude_default: True to exclude the default modifiers from the stage,
        False otherwise
    :param args_evaluated: the evaluated RecipeArgs for the stage
    """

    group: Optional[str] = None
    args: Optional[RecipeArgs] = None
    enabled: bool = True
    modifiers: List[RecipeModifier] = Field(default_factory=list)
    exclude_default: bool = False
    args_evaluated: Optional[RecipeArgs] = None

    def calculate_start(self) -> int:
        """
        :return: the start epoch for the stage, atleast one modifier
            in current stage must have a start
        """
        return min(
            mod.calculate_start()
            for mod in self.modifiers
            if mod.calculate_start() >= 0
        )

    def calculate_end(self) -> int:
        """
        :return: the end epoch for the stage, atleast one modifier
            in current stage must have an end
        """
        return max(
            mod.calculate_end() for mod in self.modifiers if mod.calculate_end() >= 0
        )

    def evaluate(
        self, parent_args: Optional[RecipeArgs] = None, shift: Optional[int] = None
    ):
        """
        Evaluate the args for the stage with parent_args if any and shift
        the start and end if provided

        :param parent_args: Optional RecipeArgs to use for evaluation
        :param shift: Optional amount to shift the start and end by,
            defaults to None (no shift)
        """
        if self.args is None:
            self.args = RecipeArgs({})
        merged_args = self.args.combine(parent_args)
        self.args_evaluated = merged_args.evaluate()
        for modifier in self.modifiers:
            modifier.evaluate(self.args_evaluated, shift)

    def create_modifier(
        self, framework: Framework, parent_args: RecipeArgs = None
    ) -> StageModifiers:
        """
        Evaluate curent stage with parent_args if any and return
        StageModifiers instance.

        The StageModifiers instance will contain instantiated Framework
        specific modifiers for the stage with the group and index set

        evaluate(...)
            | evaluate stage with parent_args if any
            | for each recipe_modifier in stage
            |   | instantiate Framework specific modifier
            |   | set group and index of modifier
            |   | append modifier to StageModifiers.modifiers
            | return StageModifiers instance

        :param framework: the framework to create the modifiers for
        :param parent_args: Optional RecipeArgs to use for evaluation
        :return: the StageModifiers for the stage
        """
        if parent_args is not None:
            self.evaluate(parent_args)

        stage_modifiers = StageModifiers()
        for index, modifier in enumerate(self.modifiers):
            modifier = modifier.create_modifier(framework)
            modifier.group = self.group
            modifier.index = index
            stage_modifiers.modifiers.append(modifier)

        return stage_modifiers

    @root_validator(pre=True)
    def remap_modifiers(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        modifiers = RecipeStage.extract_dict_modifiers(values)
        values["modifiers"] = modifiers

        return values

    @staticmethod
    def extract_dict_modifiers(values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts modifiers from a dict of values and returns a list of modifiers
        with the group set to the key of the modifier in the dict

        >>> values = {
        ...     "pruning_modifiers": {
        ...         "ModifierTypeOne": {"param": 1},
        ...         "ModifierTypeTwo": {"param": 2},
        ...     },
        ... }
        >>> RecipeStage.extract_dict_modifiers(values) # doctest: +NORMALIZE_WHITESPACE
        [{'ModifierTypeOne': {'param': 1}, 'group': 'pruning'},
        {'ModifierTypeTwo': {'param': 2}, 'group': 'pruning'}]

        Accepted formats:
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

        modifiers = []
        remove_keys = []

        if "modifiers" in values and values["modifiers"]:
            remove_keys.append("modifiers")
            for mod_key, mod_value in values["stages"].items():
                modifier = {mod_key: mod_value}
                modifier["group"] = "default"
                modifiers.append(modifier)

        for key, value in list(values.items()):
            if key.endswith("_modifiers"):
                remove_keys.append(key)
                group = key.rsplit("_modifiers", 1)[0]
                for mod_key, mod_value in value.items():
                    modifier = {mod_key: mod_value}
                    modifier["group"] = group
                    modifiers.append(modifier)

        for key in remove_keys:
            del values[key]

        return modifiers

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        :return: a dictionary representation of the stage
        """
        dict_ = super().dict(*args, **kwargs)
        modifiers = {}

        for modifier in dict_["modifiers"]:
            group = modifier["group"]
            del modifier["group"]
            if group not in modifiers:
                modifiers[group] = []
            modifiers[group].append(modifier)

        dict_["modifiers"] = modifiers

        return dict_
