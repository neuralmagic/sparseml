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
from typing import Any, Dict, List, Tuple, Union

from pydantic import Field

from sparseml.core.data import ModifiableData
from sparseml.core.event import Event, EventLifecycle
from sparseml.core.framework import Framework
from sparseml.core.model import ModifiableModel
from sparseml.core.optimizer import ModifiableOptimizer
from sparseml.core.recipe import Recipe


__all__ = ["State", "Data", "Hardware", "ModifiedState"]


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
    recipes: List[Tuple[Recipe, str, Dict[str, Any]]] = Field(default_factory=list)
    loggers = Field(default_factory=list)
    framework: Framework = None
    model: ModifiableModel = None
    optimizer: ModifiableOptimizer = None
    optim_wrapped: bool = None
    loss = None
    batch_data = None
    data = Data()
    hardware = Hardware()
    event_lifecycle: EventLifecycle = None
    last_event: Event = None

    def update_framework(self, framework: Framework):
        self.framework = framework if framework else Framework.pytorch

    def update_recipe(
        self,
        recipe: Union[str, List[str], Recipe, List[Recipe]] = None,
        recipe_stage: str = None,
        recipe_args: Dict[str, Any] = None,
    ):
        pass

    def update_model(self, model: Any):
        pass

    def update_optimizer(self, optimizer: Any, attach_callbacks: bool = True):
        pass

    def update_data(
        self,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        calib_data: Any = None,
        copy_data: bool = True,
    ):
        pass

    def update_start(
        self,
        start: float = None,
        steps_per_epoch: int = None,
        batches_per_step: int = None,
    ):
        pass

    def should_recompile_recipe(self) -> bool:
        pass

    def recompile_recipe(self) -> Recipe:
        pass


@dataclass
class ModifiedState:
    model: Any = None
    optimizer: Any = None
    loss: Any = None
    modifier_data: List[Dict[str, Any]] = None

    def __init__(self, model, optimizer, loss, modifier_data):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.modifier_data = modifier_data
