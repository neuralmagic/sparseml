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

from enum import Enum
from dataclasses import dataclass


__all__ = ["EventType", "Event"]


class EventType(Enum):
    # training lifecycle
    PRE_INIT = "pre_init"
    INITIALIZE = "initialize"
    FINALIZE = "finalize"

    # step lifecycle
    BATCH_START = "batch_start"
    LOSS_CALCULATED = "loss_calculated"
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"
    BATCH_END = "batch_end"


@dataclass
class Event:
    type_: EventType = EventType.PRE_INIT

    epoch_based: bool = None
    steps_per_epoch: int = None
    batches_per_step: int = None

    global_step: int = None
    global_batch: int = None
    epoch: int = None
    epoch_step: int = None
    epoch_batch: int = None

    def current_index(self) -> float:
        if not self.epoch_based:
            return self.global_step

        epoch = self.epoch + (self.epoch_step / self.steps_per_epoch)

        if epoch - self.epoch > 1.0:
            raise ValueError("too many steps per epoch for epoch based event")

        return epoch
