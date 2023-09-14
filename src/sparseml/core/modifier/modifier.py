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


from typing import Optional

from pydantic import BaseModel

from sparseml.core.event import Event, EventType
from sparseml.core.framework import MultiFrameworkObject
from sparseml.core.modifier.base import ModifierInterface

__all__ = ["Modifier"]


class Modifier(ModifierInterface, MultiFrameworkObject, BaseModel):
    index: int = None
    group: str = None
    start: float = None
    end: Optional[float] = None
    update: Optional[float] = None

    _initialized_structure: bool = False
    _initialized: bool = False
    _finalized: bool = False
    _started: bool = False
    _ended: bool = False

    def check_initialized(self):
        if not self._initialized:
            raise RuntimeError("modifier has not been initialized")

    def calculate_start(self) -> float:
        return self.start if self.start is not None else -1

    def calculate_end(self) -> float:
        return self.end if self.end is not None else -1

    def pre_initialize_structure(self, state: "State", **kwargs):
        self.on_initialize_structure(state, **kwargs)
        self._initialized_structure = True

    def initialize(self, state: "State", **kwargs):
        if self._initialized:
            return

        if self._finalized:
            raise RuntimeError("cannot initialize a finalized modifier")

        if state.start_event is None:
            return

        initialized = self.on_initialize(**kwargs)

        if not isinstance(initialized, bool):
            raise ValueError(
                "on_initialize must return a boolean value; "
                "True for success, False for not initialized"
            )

        self._initialized = initialized

        if self.should_start(state.start_event):
            self.on_start(state, state.start_event, **kwargs)
            self._started = True

    def finalize(self, state: "State", **kwargs):
        if self._finalized:
            return

        if not self._initialized:
            raise RuntimeError("cannot finalize an uninitialized modifier")

        finalized = self.on_finalize(**kwargs)

        if not isinstance(finalized, bool):
            raise ValueError(
                "on_finalize must return a boolean value; "
                "True for success, False for not finalized"
            )

        self._finalized = finalized

    def update_event(self, state: "State", event: Event, **kwargs):
        if not self._initialized:
            raise RuntimeError("cannot update an uninitialized modifier")

        if self._finalized:
            raise RuntimeError("cannot update a finalized modifier")

        # handle starting the modifier if needed
        if (
            event.type_ == EventType.BATCH_START
            and not self._started
            and self.should_start(event)
        ):
            self.on_start(state, event, **kwargs)
            self._started = True
            self.on_update(state, event, **kwargs)

            return

        # handle ending the modifier if needed
        if (
            event.type_ == EventType.BATCH_END
            and not self._ended
            and self.should_end(event)
        ):
            self.on_end(state, event, **kwargs)
            self._ended = True
            self.on_update(state, event, **kwargs)

            return

        if self._started and not self._ended:
            self.on_update(state, event, **kwargs)

    def should_start(self, event: Event):
        current = event.current_index

        return self.start <= current and (self.end is None or current < self.end)

    def should_end(self, event: Event):
        current = event.current_index

        return self.end is not None and current >= self.end

    def on_initialize_structure(self, state: "State", **kwargs):
        raise NotImplementedError()

    def on_initialize(self, state: "State", event: Event, **kwargs) -> bool:
        raise NotImplementedError()

    def on_finalize(self, state: "State", event: Event, **kwargs) -> bool:
        raise NotImplementedError()

    def on_start(self, state: "State", event: Event, **kwargs):
        raise NotImplementedError()

    def on_update(self, state: "State", event: Event, **kwargs):
        raise NotImplementedError()

    def on_end(self, state: "State", event: Event, **kwargs):
        raise NotImplementedError()
