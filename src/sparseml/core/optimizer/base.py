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

from pydantic import BaseModel
from typing import TypeVar, Generic, Union, List, Any

from sparseml.core.framework import MultiFrameworkObject

__all__ = ["ModifiableOptimizer"]


OT = TypeVar("OT")
PGT = TypeVar("PGT")


class ModifiableOptimizer(Generic[OT, PGT], MultiFrameworkObject, BaseModel):
    optimizer: OT = None

    def get_param_groups(self) -> List[PGT]:
        raise NotImplementedError()

    def set_param_groups(self, param_groups: List[PGT]):
        raise NotImplementedError()

    def get_learning_rate(
        self, group_index: Union[int, None] = None
    ) -> Union[float, List[float]]:
        raise NotImplementedError()

    def set_learning_rate(self, lr: float, group_index: Union[int, None] = None):
        raise NotImplementedError()

    def get_attribute(
        self, name: str, group_index: Union[int, None] = None
    ) -> Union[Any, List[Any]]:
        raise NotImplementedError()

    def set_attribute(
        self, name: str, value: Any, group_index: Union[int, None] = None
    ):
        raise NotImplementedError()
