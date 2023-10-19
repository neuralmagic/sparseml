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
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import Field, root_validator

from sparseml.core.framework import Framework
from sparseml.core.modifier import StageModifiers
from sparseml.core.recipe.args import RecipeArgs
from sparseml.core.recipe.base import RecipeBase
from sparseml.core.recipe.metadata import RecipeMetaData
from sparseml.core.recipe.stage import RecipeStage


__all__ = ["Recipe", "RecipeTuple"]


class Recipe(RecipeBase):
    """
    A class to represent a recipe for a model.
    Recipes encode the instructions needed for modifying
    the model and/or training process as a list of modifiers.
    (More information on supported modifiers can be found at
    https://docs.neuralmagic.com/products/sparseml)

    Recipes can be created from a file, string, or SparseZoo stub.
    Acceptable file formats include both json and yaml, however,
    when serializing a recipe, yaml will be used by default.
    """

    @staticmethod
    def create_instance(path: str) -> "Recipe":
        """
        Create a recipe instance from a file, or string


        Using a recipe string or file is supported:
        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str)

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
        recipe: Union["Recipe", "RecipeTuple"], shift: Optional[int] = None
    ) -> "Recipe":
        """
        Simplify a RecipeTuple by removing stages that are not in the target_stages
        and shifting the start and end of the recipe by the shift amount


        Using a RecipeTuple instance with shift:
        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str)
        >>> recipe_tuple = RecipeTuple(recipe, ["test"], {})
        >>> simplified = Recipe.simplify_recipe(recipe_tuple, shift=2)
        >>> simplified.stages[0].modifiers[0].args_evaluated["start"]
        2.0

        :param recipe: The Recipe or RecipeTuple instance to simplify
        :param shift: The amount to shift the start and end of the recipe by,
            defaults to None (No shift)
        :return: The simplified Recipe instance
        """
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
        """
        A method to combine multiple recipes into one recipe
        Automatically calculates the start and end of the combined recipe
        and shifts the start and end of the recipes accordingly

        Using two RecipeTuple instances:
        >>> recipe_str_1 = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe_str_2 = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 3.0
        ...             end: 5.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe_1, recipe_2 = (Recipe.create_instance(recipe_str_1),
        ... Recipe.create_instance(recipe_str_2))
        >>> combined = Recipe.simplify_combine_recipes(
        ... [RecipeTuple(recipe_1, ["test"], {}), RecipeTuple(recipe_2, ["test"], {})])
        >>> len(combined.stages)
        2

        :param recipes: The list of Recipe/RecipeTuple instances to combine
        :return: The combined Recipe instance
        """

        combined = Recipe()

        for recipe in recipes:
            simplified = Recipe.simplify_recipe(
                recipe=recipe,
                shift=combined.calculate_end(),
            )
            combined.version = simplified.version
            combined.stages.extend(simplified.stages)
            combined.args.update(simplified.args)

        return combined

    version: str = None
    args: RecipeArgs = Field(default_factory=RecipeArgs)
    stages: List[RecipeStage] = Field(default_factory=list)
    metadata: RecipeMetaData = None
    args_evaluated: RecipeArgs = Field(default_factory=RecipeArgs)

    def calculate_start(self) -> int:
        """
        Calculate and return the start epoch of the recipe.
        The start epoch is the minimum start epoch of all stages.
        Must have at least one stage to calculate the start epoch

        :return: The start epoch of the stage
        """
        return min(
            stage.calculate_start()
            for stage in self.stages
            if stage.calculate_start() >= 0
        )

    def calculate_end(self) -> int:
        """
        Calculate and return the end epoch of the recipe.
        The end epoch is the maximum end epoch of all stages.

        :return: The end of the recipe, the maximum end of all stages. If no stages
            found, returns 0
        """
        if len(self.stages) == 0:
            return 0
        return max(
            stage.calculate_end() for stage in self.stages if stage.calculate_end() >= 0
        )

    def evaluate(
        self, args: Optional[Dict[str, Any]] = None, shift: Optional[int] = None
    ):
        """
        Evaluate the recipe by evaluating all stages and combining the args
        with existing recipe_args

        Evaluate with no shift:
        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: eval(start_epoch)
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str)
        >>> recipe.evaluate({"start_epoch": 1})
        >>> recipe.stages[0].modifiers[0].args_evaluated["start"]
        1.0

        Evaluate with shift:
        >>> recipe.evaluate({"start_epoch": 2}, shift=2)
        >>> recipe.stages[0].modifiers[0].args_evaluated["start"]
        4.0

        :param args: The args to evaluate the recipe with
        :param shift: The amount to shift the start and end of the recipe by,
            defaults to None (No shift)
        """
        args = self.args.combine(args) if self.args else RecipeArgs(**(args or {}))
        self.args_evaluated = args.evaluate()
        for stage in self.stages:
            stage.evaluate(self.args_evaluated, shift)

    def create_modifier(self, framework: Framework) -> List["StageModifiers"]:
        """
        Create and return a list of StageModifiers for each stage in the recipe

        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str)
        >>> stage_modifiers = recipe.create_modifier(Framework.pytorch)
        >>> len(stage_modifiers) == 1
        True
        >>> len(stage_modifiers[0].modifiers) == 1
        True

        :param framework: The framework to create the modifiers for
        :return: A list of StageModifiers for each stage in the recipe
        """
        if not self.args_evaluated:
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
        Extract stages from a dict of values, acceptable dictionary structures
        are shown below

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

        >>> values = {
        ... "stages": {
        ...     "first_stage": {
        ...         "modifiers": {
        ...             "ModifierTypeOne": {
        ...                 "start": 0.0,
        ...                 "end": 2.0,
        ...                 }
        ...         }
        ...     }
        ... }
        ... }
        >>> Recipe.extract_dict_stages(values) # doctest: +NORMALIZE_WHITESPACE
        [{'modifiers': {'ModifierTypeOne': {'start': 0.0, 'end': 2.0}},
        'group': 'first_stage'}]

        :param values: The values dict to extract stages from
        :return: A list of stages, where each stage is a dict of
            modifiers and their group
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
        """
        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str)
        >>> recipe.dict()
        Traceback (most recent call last):
        ...
        KeyError: 'group'

        :return: A dictionary representation of the recipe
        """
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
    """
    A simple dataclass to hold a recipe, it's target_stages, and override_args

    :param recipe: The Recipe instance to hold
    :param target_stages: The stages to target when simplifying the recipe
        (Note: Stages not in the target_stages will be removed during
        simplification)
    :param override_args: The args used to override existing recipe args
        associated with the supplied `recipe`
    """

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
