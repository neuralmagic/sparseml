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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from sparseml.core.recipe.recipe import Recipe, RecipeTuple


__all__ = ["RecipeContainer"]


@dataclass
class RecipeContainer:
    """
    A container for recipes to be used in a session. Provides utilities
    to update the recipes and compile them into a single recipe.

    :param compiled_recipe: the compiled recipe from the recipes list
    :param recipes: the list of RecipeTuple instances to be compiled
    :param applied_stages: list of recipe stages that have already been applied
    """

    compiled_recipe: Optional[Recipe] = None
    recipes: List[RecipeTuple] = field(default_factory=list)
    applied_stages: List[str] = field(default_factory=list)

    def update(
        self,
        recipe: Union[str, List[str], Recipe, List[Recipe], None] = None,
        recipe_stage: Union[str, List[str], List[List[str]], None] = None,
        recipe_args: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        **kwargs,
    ) -> Dict:
        """
        Update the recipes in the container. If a recipe is provided, it will
        reset any existing compiled_recipe in the container. Must call
        `check_compile_recipe` to re-compile the recipes into a single compiled_recipe.
        If no recipe is provided, does nothing and returns the kwargs.

        Can provide multiple recipes to update the container with:
        >>> container = RecipeContainer()
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
        ...             end: 4.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> result = container.update(recipe=[recipe_str_1, recipe_str_2])

        :param recipe: the recipe to update the container with
        :param recipe_stage: the recipe stage to update the container with
        :param recipe_args: the recipe args to update the recipe with
        :param kwargs: additional kwargs to return
        :return: the passed in kwargs
        """
        if recipe is not None:
            self.compiled_recipe = None

            if not isinstance(recipe, list):
                recipe = [recipe]
            if recipe_stage is None:
                recipe_stage = [None] * len(recipe)
            else:
                if not isinstance(recipe_stage, list):
                    recipe_stage = [[recipe_stage]] * len(recipe)
                if not isinstance(recipe_stage[0], list):
                    recipe_stage = [recipe_stage] * len(recipe)

            if recipe_args is None:
                recipe_args = [{}] * len(recipe)
            elif not isinstance(recipe_args, list):
                recipe_args = [recipe_args] * len(recipe)

            if len(recipe) != len(recipe_stage) or len(recipe) != len(recipe_args):
                raise ValueError(
                    "recipe, recipe_stage, and recipe_args must be the same length"
                )

            for rec, stage, args in zip(recipe, recipe_stage, recipe_args):
                if isinstance(rec, str):
                    rec = Recipe.create_instance(rec)
                self.recipes.append(RecipeTuple(rec, stage, args))

        return kwargs

    def update_applied_stages(self, new_stages: List[str]):
        """
        Updates the applied_stages list with new stages, indicating their structure
        has already been applied

        :param new_stages: new stage names to add
        """
        for stage in new_stages:
            if stage not in self.applied_stages:
                self.applied_stages.append(stage)

    def check_compile_recipe(self) -> bool:
        """
        Check if the recipes need to be compiled into a single recipe and
        compile them if they do.

        :return: True if the recipes were compiled, False otherwise
        """
        if self.compiled_recipe is None and self.recipes:
            self.compiled_recipe = Recipe.simplify_combine_recipes(self.recipes)
            return True

        return False

    def check_any_recipe_exists(self) -> bool:
        """
        Checks if any recipes have been added to the container, compiled or not

        :return: True if any recipes exist in the container, False otherwise
        """
        if self.compiled_recipe is not None:
            return True
        if len(self.recipes) > 0:
            return True

        return False
