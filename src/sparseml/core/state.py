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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import Field

from sparseml.core.data import ModifiableData
from sparseml.core.event import Event
from sparseml.core.framework import Framework
from sparseml.core.model import ModifiableModel
from sparseml.core.optimizer import ModifiableOptimizer


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
    framework: Framework
    model: ModifiableModel = None
    teacher_model: ModifiableModel = None
    optimizer: ModifiableOptimizer = None
    optim_wrapped: bool = None
    loss: Any = None
    batch_data: Any = None
    data = Data()
    hardware = Hardware()
    start_event: Event = None
    last_event: Event = None
    loggers = Field(default_factory=list)

    @property
    def sparsification_ready(self) -> bool:
        return (
            self.model is not None
            and self.optimizer is not None
            # and self.loss is not None
            # and self.batch_data is not None
        )

    def update(
        self,
        model: Any = None,
        teacher_model: Any = None,
        optimizer: Any = None,
        attach_optim_callbacks: bool = True,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        calib_data: Any = None,
        copy_data: bool = True,
        start: float = None,
        steps_per_epoch: int = None,
        batches_per_step: int = None,
        **kwargs,
    ) -> Dict:
        if model is not None:
            self.model = ModifiableModel(framework=self.framework, model=model)
        if teacher_model is not None:
            self.teacher_model = ModifiableModel(
                framework=self.framework, model=teacher_model
            )
        if optimizer is not None:
            self.optim_wrapped = attach_optim_callbacks
            self.optimizer = ModifiableOptimizer(
                framework=self.framework, optimizer=optimizer
            )

        if train_data is not None:
            self.data.train = train_data if not copy_data else deepcopy(train_data)
        if val_data is not None:
            self.data.val = val_data if not copy_data else deepcopy(val_data)
        if test_data is not None:
            self.data.test = test_data if not copy_data else deepcopy(test_data)
        if calib_data is not None:
            calib_loader = calib_data if not copy_data else deepcopy(calib_data)
            self.calib_data = ModifiableData(
                framework=self.framework, data_loader=calib_loader
            )

        if (
            start is not None
            or steps_per_epoch is not None
            or batches_per_step is not None
        ):
            if self.start_event is None:
                self.start_event = Event()

            if start is not None:
                self.start_event.current_index = start
            if steps_per_epoch is not None:
                self.start_event.steps_per_epoch = steps_per_epoch
            if batches_per_step is not None:
                self.start_event.batches_per_step = batches_per_step

        return kwargs


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
