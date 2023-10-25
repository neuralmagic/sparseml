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

from sparseml.core.event import Event
from sparseml.core.state import State


__all__ = ["ModifierInterface"]


class ModifierInterface(ABC):
    """
    Defines the contract that all modifiers must implement
    """

    @property
    @abstractmethod
    def initialized_structure(self) -> bool:
        """
        :return: True if the modifier structure has been
            applied to the model
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """
        :return: True if the modifier has been initialized
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def finalized(self) -> bool:
        """
        :return: True if the modifier has been finalized
        """
        raise NotImplementedError()

    @abstractmethod
    def check_initialized(self):
        """
        Check if the modifier has been initialized and
        raise an error if not
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_start(self) -> float:
        """
        :return: the start step for the modifier
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_end(self) -> float:
        """
        :return: the end step for the modifier
        """
        raise NotImplementedError()

    @abstractmethod
    def pre_initialize_structure(self, state: State, **kwargs):
        """
        Apply the modifier structure to the model

        :param state: The current state of the model
        :param kwargs: Additional arguments for the modifier
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, state: State, **kwargs):
        """
        Initialize the modifier

        :param state: The current state of the model
        :param kwargs: Additional keyword arguments
            for modifier initialization
        """
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, state: State, **kwargs):
        """
        Finalize the modifier

        :param state: The current state of the model
        :param kwargs: Additional keyword arguments for
            modifier finalization
        """
        raise NotImplementedError()

    @abstractmethod
    def update_event(self, state: State, event: Event, **kwargs):
        """
        Update the modifier based on the event

        :param state: The current state of the model
        :param event: The event to update the modifier with
        :param kwargs: Additional keyword arguments for
            modifier update
        """
        raise NotImplementedError()
