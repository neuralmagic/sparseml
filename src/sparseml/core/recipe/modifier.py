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

from typing import Dict, Any
from pydantic import BaseModel, root_validator

from sparseml.core.recipe.args import RecipeArgs
from sparseml.core.modifier import Modifier, ModifierFactory
from sparseml.core.framework import Framework


__all__ = ["RecipeModifier"]


class RecipeModifier(BaseModel):
    type: str
    group: str = None
    args: Dict[str, Any] = None
    _args: Dict[str, Any] = None

    def evaluate(self, parent_args: RecipeArgs):
        if self.args:
            self._args = parent_args.evaluate_ext(self.args)
        else:
            self._args = dict()

    def create_modifier(self, framework: Framework) -> Modifier:
        return ModifierFactory.create(self.type, framework, **self._args)

    @root_validator(pre=True)
    def extract_modifier_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        assert len(values) == 1, "multiple key pairs found for modifier"
        modifier_type, args = list(values.items())[0]

        return {"type": modifier_type, "args": args}

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        return {self.type: self.args}
