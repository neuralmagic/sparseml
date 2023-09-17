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
    compiled_recipe: Optional[Recipe] = None
    recipes: List[RecipeTuple] = field(default_factory=list)

    def update(
        self,
        recipe: Union[str, List[str], Recipe, List[Recipe]] = None,
        recipe_stage: Union[str, List[str]] = None,
        recipe_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict:
        if recipe is not None:
            self.compiled_recipe = None

            if not isinstance(recipe, list):
                recipe = [recipe]

            if recipe_stage is None:
                recipe_stage = [None] * len(recipe)
            elif not isinstance(recipe_stage, list):
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

    def check_compile_recipe(self) -> bool:
        if self.compiled_recipe is None and self.recipes:
            self.compiled_recipe = Recipe.simplify_combine_recipes(self.recipes)

            return True

        return False
