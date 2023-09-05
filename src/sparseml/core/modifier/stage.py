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


from typing import List

from pydantic import BaseModel, Field

from sparseml.core.modifier.base import ModifierInterface
from sparseml.core.modifier.modifier import Modifier
from sparseml.core.state import Event, State


class StageModifiers(ModifierInterface, BaseModel):
    modifiers: List[Modifier] = Field(default_factory=list)
    index: int = None
    group: str = None

    _initialized_structure: bool = False
    _initialized: bool = False
    _finalized: bool = False

    def pre_initialize_structure(self, state: State, **kwargs):
        for modifier in self.modifiers:
            modifier.pre_initialize_structure(state, **kwargs)
        self._initialized_structure = True

    def initialize(self, state: State, **kwargs):
        for modifier in self.modifiers:
            modifier.initialize(state, **kwargs)
        self._initialized = True

    def finalize(self, state: State, **kwargs):
        for modifier in self.modifiers:
            modifier.finalize(state, **kwargs)
        self._finalized = True

    def update_event(self, state: State, event: Event, **kwargs):
        for modifier in self.modifiers:
            modifier.update_event(state, event, **kwargs)
