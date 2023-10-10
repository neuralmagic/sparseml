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


from typing import List, Optional

from pydantic import BaseModel, Field

from sparseml.core.event import Event
from sparseml.core.modifier.base import ModifierInterface
from sparseml.core.modifier.modifier import Modifier
from sparseml.core.state import State


__all__ = ["StageModifiers"]


class StageModifiers(ModifierInterface, BaseModel):
    """
    Represents a collection of modifiers that are applied together as a stage.

    :param modifiers: The modifiers to apply as a stage
    :param index: The index of the stage, if applicable
    :param group: The group name of the stage, if applicable
    """

    modifiers: List["Modifier"] = Field(default_factory=list)
    index: Optional[int] = None
    group: Optional[str] = None

    @property
    def initialized_structure(self) -> bool:
        """
        :return: True if any of the stage modifiers have initialized structure,
            False otherwise
        """
        return any(mod.initialized_structure for mod in self.modifiers)

    @property
    def initialized(self) -> bool:
        """
        :return: True if all of the stage modifiers have been initialized,
            False otherwise
        """
        return all(mod.initialized for mod in self.modifiers)

    @property
    def finalized(self) -> bool:
        """
        :return: True if all of the stage modifiers have been finalized,
            False otherwise
        """
        return all(mod.finalized for mod in self.modifiers)

    def check_initialized(self):
        """
        Check if all of the stage modifiers have been initialized,
        raises an exception if not
        """

        for modifier in self.modifiers:
            modifier.check_initialized()

    def calculate_start(self) -> float:
        """
        :return: The minimum start time of all the stage modifiers
        """
        return min(
            mod.calculate_start()
            for mod in self.modifiers
            if mod.calculate_start() >= 0
        )

    def calculate_end(self) -> float:
        """
        :return: The maximum end time of all the stage modifiers
        """
        return max(
            mod.calculate_end() for mod in self.modifiers if mod.calculate_end() >= 0
        )

    def pre_initialize_structure(self, state: "State", **kwargs):
        """
        Pre initialize the structure for all stage modifiers

        :param state: The current state of the training
        :param kwargs: Additional kwargs to pass to the modifier(s)
            pre_initialize_structure method
        """
        for modifier in self.modifiers:
            modifier.pre_initialize_structure(state, **kwargs)

    def initialize(self, state: "State", **kwargs):
        """
        Initialize all the stage modifiers

        :param state: The state of current session
        :param kwargs: Additional kwargs to pass to the modifier(s)
            initialize method
        """
        for modifier in self.modifiers:
            modifier.initialize(state, **kwargs)

    def finalize(self, state: "State", **kwargs):
        """
        Finalize all the stage modifiers

        :param state: The state of current session
        :param kwargs: Additional kwargs to pass to the modifier(s)
            finalize method
        """
        for modifier in self.modifiers:
            modifier.finalize(state, **kwargs)

    def update_event(self, state: "State", event: "Event", **kwargs):
        """
        Propagate the event to all the stage modifiers

        :param state: The state of current session
        :param event: The event to propagate
        :param kwargs: Additional kwargs to pass to the modifier(s)
            update_event method
        """
        for modifier in self.modifiers:
            modifier.update_event(state, event, **kwargs)
