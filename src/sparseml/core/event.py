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
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List


__all__ = [
    "EventType",
    "Event",
    "EventLifecycle",
    "WrappedOptimEventLifecycle",
    "CallbacksEventLifecycle",
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

    def new_instance(self, **kwargs) -> "Event":
        instance = deepcopy(self)
        for key, value in kwargs.items():
            setattr(instance, key, value)

        return instance


class EventLifecycle(ABC, Event):
    type_first: EventType = None
    step_count: int = 0
    batch_count: int = 0

    def __init__(self, type_first: EventType, start: Event):
        self.type_first = type_first
        self.steps_per_epoch = start.steps_per_epoch
        self.batches_per_step = start.batches_per_step
        self.invocations_per_step = start.invocations_per_step
        self.global_step = start.global_step
        self.global_batch = start.global_batch

    def events_from_type(self, type_: EventType) -> List[Event]:
        if type_ == EventType.BATCH_START:
            return self.batch_start_events()

        if type_ == EventType.LOSS_CALCULATED:
            return self.loss_calculated_events()

        if type_ == EventType.OPTIM_PRE_STEP:
            return self.optim_pre_step_events()

        if type_ == EventType.OPTIM_POST_STEP:
            return self.optim_post_step_events()

        if type_ == EventType.BATCH_END:
            return self.batch_end_events()

        raise ValueError(f"invalid event type {type_}")

    def check_step_batches_count(self, increment: bool) -> bool:
        if self.batches_per_step is None or self.batches_per_step < 2:
            return True

        compare_batch = self.batch_count + 1
        at_step = compare_batch % self.batches_per_step == 0

        if increment:
            self.batch_count = compare_batch if not at_step else 0

        return at_step

    def check_step_invocations_count(self, increment: bool) -> bool:
        if self.invocations_per_step is None or self.invocations_per_step < 2:
            return True

        compare_step = self.step_count + 1
        at_step = compare_step % self.invocations_per_step == 0

        if increment:
            self.step_count = compare_step if not at_step else 0

        return at_step

    @abstractmethod
    def batch_start_events(self) -> List[Event]:
        raise NotImplementedError()

    @abstractmethod
    def loss_calculated_events(self) -> List[Event]:
        raise NotImplementedError()

    @abstractmethod
    def optim_pre_step_events(self) -> List[Event]:
        raise NotImplementedError()

    @abstractmethod
    def optim_post_step_events(self) -> List[Event]:
        raise NotImplementedError()

    @abstractmethod
    def batch_end_events(self) -> List[Event]:
        raise NotImplementedError()


class WrappedOptimEventLifecycle(EventLifecycle):
    """
    Optimizer is wrapped and no batch or optim callbacks
        - batch_start: must not be invoked, auto triggered
          from loss calculated if that is called, otherwise from pre_step
        - loss_calculated: must be called before batch_end and optim_pre_step
        - batch_end: must not be invoked, auto triggered from optim_post_step
        - optim_pre_step: must be called before optim_post_step
        - optim_post_step: must be called only once after optim_pre_step
    """

    def batch_start_events(self) -> List[Event]:
        raise ValueError("batch start should not be invoked when only wrapped optim")

    def loss_calculated_events(self) -> List[Event]:
        if self.type_first != EventType.LOSS_CALCULATED:
            raise ValueError("loss calculated must be called first for wrapped optim")

        if (
            self.type_ != EventType.OPTIM_POST_STEP
            and self.type_ != EventType.LOSS_CALCULATED
        ):
            raise ValueError(
                "loss calculated must be called after batch end or optim post step"
            )

        self.type_ = EventType.LOSS_CALCULATED
        self.global_batch += 1

        if not self.check_step_batches_count(increment=True):
            # step won't be called, so batch end must be called
            return [
                self.new_instance(type_=EventType.BATCH_START),
                self.new_instance(type_=EventType.LOSS_CALCULATED),
                self.new_instance(type_=EventType.BATCH_END),
            ]
        else:
            # batch end handled by optim step
            return [
                self.new_instance(type_=EventType.BATCH_START),
                self.new_instance(type_=EventType.LOSS_CALCULATED),
            ]

    def optim_pre_step_events(self) -> List[Event]:
        if (
            self.type_first == EventType.OPTIM_PRE_STEP
            and self.type_ is not None
            and self.type_ != EventType.OPTIM_POST_STEP
        ):
            raise ValueError("optim pre step must be called after optim post step")

        if (
            self.type_first == EventType.LOSS_CALCULATED
            and self.type_ != EventType.LOSS_CALCULATED
        ):
            raise ValueError("optim pre step must be called after loss calculated")

        self.type_ = EventType.OPTIM_PRE_STEP

        if self.type_first == EventType.OPTIM_PRE_STEP:
            self.global_batch += (
                1
                if self.batches_per_step is None or self.batches_per_step < 2
                else self.batches_per_step
            )
            batch_start_events = [self.new_instance(type_=EventType.BATCH_START)]
        else:
            batch_start_events = []

        if not self.check_step_invocations_count(increment=False):
            return batch_start_events

        return batch_start_events + [
            self.new_instance(type_=EventType.OPTIM_PRE_STEP),
        ]

    def optim_post_step_events(self) -> List[Event]:
        if self.type_ != EventType.OPTIM_PRE_STEP:
            raise ValueError("optim post step must be called after optim pre step")

        self.type_ = EventType.OPTIM_POST_STEP

        if not self.check_step_invocations_count(increment=True):
            return [
                self.new_instance(type_=EventType.BATCH_END),
            ]

        self.global_step += 1

        return [
            self.new_instance(type_=EventType.OPTIM_POST_STEP),
            self.new_instance(type_=EventType.BATCH_END),
        ]

    def batch_end_events(self) -> List[Event]:
        raise ValueError("batch end should not be invoked when only wrapped optim")


class CallbacksEventLifecycle(EventLifecycle):
    """
    Optimizer is not wrapped, callbacks must be used
        - batch_start: must be called first
        - loss_calculated: must be called before batch_end and optim_post_step
        - batch_end: must be called before next batch start
        - optim_pre_step: must be invoked before optim_post_step
        - optim_post_step: must be called only once after optim_pre_step
    """

    def batch_start_events(self) -> List[Event]:
        if self.type_first != EventType.BATCH_START:
            raise ValueError("batch start must be called first for callbacks")

        if self.type_ is not None and self.type_ != EventType.BATCH_END:
            raise ValueError("batch start must be called after batch end")

        self.type_ = EventType.BATCH_START
        self.global_batch += 1

        return [self.new_instance(type_=EventType.BATCH_START)]

    def loss_calculated_events(self) -> List[Event]:
        if self.type_ != EventType.BATCH_START:
            raise ValueError("loss calculated must be called after batch start")

        self.type_ = EventType.LOSS_CALCULATED

        return [self.new_instance(type_=EventType.LOSS_CALCULATED)]

    def optim_pre_step_events(self) -> List[Event]:
        if (
            self.type_ != EventType.BATCH_START
            and self.type_ != EventType.LOSS_CALCULATED
        ):
            raise ValueError(
                "optim pre step must be called after batch start or loss calculated"
            )

        self.type_ = EventType.OPTIM_PRE_STEP

        if not self.check_step_invocations_count(increment=False):
            return []

        return [
            self.new_instance(type_=EventType.OPTIM_PRE_STEP),
        ]

    def optim_post_step_events(self) -> List[Event]:
        if self.type_ != EventType.OPTIM_PRE_STEP:
            raise ValueError("optim post step must be called after optim pre step")

        self.type_ = EventType.OPTIM_POST_STEP

        if not self.check_step_invocations_count(increment=True):
            return []

        self.global_step += 1

        return [
            self.new_instance(type_=EventType.OPTIM_POST_STEP),
        ]

    def batch_end_events(self) -> List[Event]:
        if (
            self.type_ != EventType.OPTIM_POST_STEP
            and self.type_ != EventType.LOSS_CALCULATED
            and self.type_ != EventType.BATCH_START
        ):
            raise ValueError(
                "batch end must be called after optim post step or "
                "loss calculated or batch start"
            )

        self.type_ = EventType.BATCH_END

        return [
            self.new_instance(type_=EventType.BATCH_END),
        ]
