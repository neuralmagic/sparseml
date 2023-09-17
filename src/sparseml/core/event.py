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
from enum import Enum
from typing import Optional


__all__ = [
    "EventType",
    "Event",
]


class EventType(Enum):
    # training lifecycle
    PRE_INIT = "pre_init"
    INITIALIZE = "initialize"
    FINALIZE = "finalize"

    # batch lifecycle
    BATCH_START = "batch_start"
    LOSS_CALCULATED = "loss_calculated"
    BATCH_END = "batch_end"

    # step lifecycle
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"

    def order(self) -> int:
        if self == EventType.PRE_INIT:
            return 0
        elif self == EventType.INITIALIZE:
            return 10
        elif self == EventType.FINALIZE:
            return 20
        elif self == EventType.BATCH_START:
            return 100
        elif self == EventType.LOSS_CALCULATED:
            return 110
        elif self == EventType.OPTIM_PRE_STEP:
            return 120
        elif self == EventType.OPTIM_POST_STEP:
            return 130
        elif self == EventType.BATCH_END:
            return 140
        else:
            raise ValueError(f"invalid event type {self}")


@dataclass
class Event:
    type_: EventType = None

    steps_per_epoch: int = None
    batches_per_step: int = None
    invocations_per_step: int = None

    global_step: int = 0
    global_batch: int = 0

    @property
    def epoch_based(self) -> bool:
        return self.steps_per_epoch is not None

    @property
    def epoch(self) -> int:
        return self.global_step // self.steps_per_epoch

    @property
    def epoch_full(self) -> float:
        return self.global_step / float(self.steps_per_epoch)

    @property
    def epoch_step(self) -> int:
        return self.global_step % self.steps_per_epoch

    @property
    def epoch_batch(self) -> int:
        batches_per_epoch = (
            self.steps_per_epoch * self.batches_per_step
            if self.batches_per_step
            else self.steps_per_epoch
        )

        return self.global_batch % batches_per_epoch

    @property
    def current_index(self) -> float:
        if not self.epoch_based:
            return self.global_step

        if self.epoch_full - self.epoch > 1.0:
            raise ValueError("too many steps per epoch for epoch based event")

        return self.epoch_full

    @current_index.setter
    def current_index(self, value: float):
        if not self.epoch_based:
            self.global_step = int(value)
            self.global_batch = (
                self.global_step
                if self.batches_per_step is None or self.batches_per_step < 2
                else self.global_step * self.batches_per_step
            )
            return

        self.global_step = int(value * self.steps_per_epoch)
        self.global_batch = (
            self.global_step
            if self.batches_per_step is None or self.batches_per_step < 2
            else self.global_step * self.batches_per_step
        )

    def should_update(
        self, start: Optional[float], end: Optional[float], update: float
    ):
        current = self.current_index

        if start is not None and current < start:
            return False

        if end is not None and current > end:
            return False

        return update is None or update <= 0.0 or current % update < 1e-10

    def new_instance(self, **kwargs) -> "Event":
        instance = deepcopy(self)
        for key, value in kwargs.items():
            setattr(instance, key, value)

        return instance
