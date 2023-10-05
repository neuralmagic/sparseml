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
    @property
    @abstractmethod
    def initialized_structure(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def initialized(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def finalized(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def check_initialized(self):
        raise NotImplementedError()

    @abstractmethod
    def calculate_start(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def calculate_end(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def pre_initialize_structure(self, state: State, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, state: State, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, state: State, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def update_event(self, state: State, event: Event, **kwargs):
        raise NotImplementedError()
