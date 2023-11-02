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
from typing import List, Optional

from sparseml.core.event import Event, EventType


__all__ = [
    "EventLifecycle",
    "WrappedOptimEventLifecycle",
    "CallbacksEventLifecycle",
]


class EventLifecycle(ABC, Event):
    """
    A lifecycle for events to be used in a SparseML session.
    Provides base utilities and also defines the contract that
    all inheritors must follow.

    The order in which the events are called is determined by
    the inheritors of this class.

    :param type_first: The first event type to be called
    :param start: The start event to base the lifecycle off of
    """

    type_first: Optional[EventType] = None
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
        """
        :param type_: The event type to get the events for
        :return: The list of events for the given type
        """
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
        """
        :return: True if the batch count is at the step count, False otherwise
        """
        if self.batches_per_step is None or self.batches_per_step < 2:
            return True

        compare_batch = self.batch_count + 1
        at_step = compare_batch % self.batches_per_step == 0

        if increment:
            self.batch_count = compare_batch if not at_step else 0

        return at_step

    def check_step_invocations_count(self, increment: bool) -> bool:
        """
        :return: True if the invocation count is at the step count, False otherwise
        """
        if self.invocations_per_step is None or self.invocations_per_step < 2:
            return True

        compare_step = self.step_count + 1
        at_step = compare_step % self.invocations_per_step == 0

        if increment:
            self.step_count = compare_step if not at_step else 0

        return at_step

    @abstractmethod
    def batch_start_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the batch start
        """
        raise NotImplementedError()

    @abstractmethod
    def loss_calculated_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the loss calculated
        """
        raise NotImplementedError()

    @abstractmethod
    def optim_pre_step_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the optim pre step
        """
        raise NotImplementedError()

    @abstractmethod
    def optim_post_step_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the optim post step
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_end_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the batch end
        """
        raise NotImplementedError()


class WrappedOptimEventLifecycle(EventLifecycle):
    """
    An event lifecycle for when the optimizer is wrapped and no batch or optimizer
    callbacks are used.
        - batch_start: must not be invoked, auto triggered
          from loss calculated if that is called, otherwise from pre_step
        - loss_calculated: must be called before batch_end and optim_pre_step
        - batch_end: must not be invoked, auto triggered from optim_post_step
        - optim_pre_step: must be called before optim_post_step
        - optim_post_step: must be called only once after optim_pre_step
    """

    def batch_start_events(self) -> List[Event]:
        """
        :raises ValueError: if invoked as this should not be called
        """
        raise ValueError("batch start should not be invoked when only wrapped optim")

    def loss_calculated_events(self) -> List[Event]:
        """
        :raises ValueError: if invoked before loss calculation
        :return: The list of events to be called for the loss calculated
        """
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
        """
        :return: The list of events to be called for the optim pre step
        """
        if (
            self.type_first == EventType.OPTIM_PRE_STEP
            and self.type_ is not None
            and self.type_ != EventType.OPTIM_POST_STEP
        ):
            raise ValueError("optim pre step must be called before optim post step")

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
        """
        :return: The list of events to be called for the optim post step
        """
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
        """
        :return: The list of events to be called for the batch end
        """
        raise ValueError("batch end should not be invoked when only wrapped optim")


class CallbacksEventLifecycle(EventLifecycle):
    """
    An event lifecycle for when the optimizer is not wrapped and callbacks are used.
        - batch_start: must be called first
        - loss_calculated: must be called before batch_end and optim_post_step
        - batch_end: must be called before next batch start
        - optim_pre_step: must be invoked before optim_post_step
        - optim_post_step: must be called only once after optim_pre_step
    """

    def batch_start_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the batch start
        """
        if self.type_first != EventType.BATCH_START:
            raise ValueError("batch start must be called first for callbacks")

        if self.type_ is not None and self.type_ != EventType.BATCH_END:
            raise ValueError("batch start must be called after batch end")

        self.type_ = EventType.BATCH_START
        self.global_batch += 1

        return [self.new_instance(type_=EventType.BATCH_START)]

    def loss_calculated_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the loss calculated
        """
        if self.type_ != EventType.BATCH_START:
            raise ValueError("loss calculated must be called after batch start")

        self.type_ = EventType.LOSS_CALCULATED

        return [self.new_instance(type_=EventType.LOSS_CALCULATED)]

    def optim_pre_step_events(self) -> List[Event]:
        """
        :return: The list of events to be called for the optim pre step
        """
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
        """
        :return: The list of events to be called for the optim post step
        """
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
        """
        :return: The list of events to be called for the batch end
        """
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
