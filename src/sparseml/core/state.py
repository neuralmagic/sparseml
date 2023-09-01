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

from dataclasses import dataclass
from typing import List
from pydantic import Field

from sparseml.core.event import Event
from sparseml.core.data import ModifiableData
from sparseml.core.model import ModifiableModel
from sparseml.core.optimizer import ModifiableOptimizer
from sparseml.core.recipe import Recipe
from sparseml.core.framework import Framework


__all__ = ["State", "Data", "Hardware"]


@dataclass
class Data:
    train: ModifiableData = None
    val: ModifiableData = None
    test: ModifiableData = None
    calib: ModifiableData = None


@dataclass
class Hardware:
    device: str = None
    devices: List[str] = None
    rank: int = None
    world_size: int = None
    local_rank: int = None
    local_world_size: int = None
    distributed: bool = None
    distributed_strategy: str = None


@dataclass
class State:
    compiled_recipe: Recipe = None
    recipes: List[Recipe] = Field(default_factory=list)
    loggers = Field(default_factory=list)
    framework: Framework = None
    model: ModifiableModel = None
    optimizer: ModifiableOptimizer = None
    data = Data()
    hardware = Hardware()
    last_event: Event = Event()
