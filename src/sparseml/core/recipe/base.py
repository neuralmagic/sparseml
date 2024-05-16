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

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from sparseml.core.framework import Framework
from sparseml.core.recipe.args import RecipeArgs


__all__ = ["RecipeBase"]


class RecipeBase(BaseModel, ABC):
    """
    Defines the contract that `Recipe` and its components
    such as `RecipeModifier` and `RecipeStage` must follow.

    All inheritors of this class must implement the following methods:
        - calculate_start
        - calculate_end
        - evaluate
        - create_modifier
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def calculate_start(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def calculate_end(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, args: Optional[RecipeArgs] = None, shift: Optional[int] = None):
        raise NotImplementedError()

    @abstractmethod
    def create_modifier(self, framework: Framework) -> Any:
        raise NotImplementedError()
