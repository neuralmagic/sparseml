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

"""
Contains base code related to modifiers: objects that modify some aspect
of the training process for a model.
For example, learning rate schedules or kernel sparsity (weight pruning)
are implemented as modifiers.
"""

from typing import List, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import (
    BaseModifier,
    BaseScheduled,
    BaseUpdate,
    ModifierProp,
    ModifierYAML,
)
from sparseml.pytorch.utils import PyTorchLogger
from sparseml.utils import ALL_TOKEN, PYTORCH_FRAMEWORK


__all__ = [
    "ModifierProp",
    "PYTORCH_FRAMEWORK",
    "PyTorchModifierYAML",
    "Modifier",
    "ScheduledModifier",
    "ScheduledUpdateModifier",
]


class PyTorchModifierYAML(ModifierYAML):
    """
    A decorator to handle making a pytorch modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.
    """

    def __init__(self):
        super().__init__(PYTORCH_FRAMEWORK)


class Modifier(BaseModifier):
    """
    The base pytorch modifier implementation,
    all modifiers must inherit from this class.
    It defines common things needed for the lifecycle and implementation of a modifier.

    | Lifecycle:
    |   - initialize
    |   - initialize_loggers
    |
    |   training loop:
    |       - update
    |       - log_update
    |       - loss_update
    |       - optimizer_pre_step
    |       - optimizer_post_step

    :param log_types: The loggers that can be used by the modifier instance
    :param kwargs: standard key word args, used to support multi inheritance
    """

    @staticmethod
    def load_list(yaml_str: str):
        """
        :param yaml_str: a string representation of the yaml syntax to
            load modifiers from
        :return: the loaded modifiers list
        """
        return Modifier.load_framework_list(yaml_str, PYTORCH_FRAMEWORK)

    @staticmethod
    def load_obj(yaml_str: str):
        """
        :param yaml_str:  a string representation of the yaml syntax to
            load a modifier from
        :return: the loaded modifier object
        """
        return Modifier.load_framework_obj(yaml_str, PYTORCH_FRAMEWORK)

    def __init__(self, log_types: Union[str, List[str]] = None, **kwargs):
        super().__init__(log_types=log_types, **kwargs)
        self._loggers_initialized = False
        self._loggers = None

    @ModifierProp(serializable=False)
    def loggers_initialized(self):
        """
        :return: True if initialize_loggers has been called, False otherwise
        """
        return self._loggers_initialized

    @ModifierProp(serializable=False)
    def loggers(self):
        """
        :return: loggers to log important info to within for this modifier
            (filtered by allowed_loggers)
        """
        return self._loggers if self._loggers is not None else []

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Handles initializing and setting up the modifier.
        Called once on construction of the scheduled optimizer.

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        self._initialized = True

    def initialize_loggers(self, loggers: Union[None, List[PyTorchLogger]]):
        """
        :param loggers: the loggers to setup this modifier with for logging important
            info and milestones to
        """
        self._loggers_initialized = True

        if not self._log_types or not loggers:
            return

        self._loggers = [
            log
            for log in loggers
            if self._log_types == ALL_TOKEN or log.name in self._log_types
        ]

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles updating the modifier's state, module, or optimizer.
        Called when update_ready() returns True.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles logging updates for the modifier for better tracking and visualization.
        Should be overwritten for logging.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._loggers_initialized:
            raise RuntimeError("modifier must have loggers initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

    def loss_update(
        self,
        loss: Tensor,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ):
        """
        Optional call that can be made on the optimizer to update the modifiers
        once the loss has been calculated.
        Called independent of if the modifier is currently active or not.

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: the modified loss tensor
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

        return loss

    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Called before the optimizer step happens
        (after backward has been called, before optimizer.step).
        Called independent of if the modifier is currently active or not.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Called after the optimizer step happens and weights have updated.
        Called independent of if the modifier is currently active or not.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")


class ScheduledModifier(Modifier, BaseScheduled):
    """
    The base scheduled modifier implementation,
    all scheduled modifiers must inherit from this class.
    The difference for this and a Modifier is that these have start and end epochs.
    It defines common things needed for the lifecycle and implementation of a
    scheduled modifier.

    | Lifecycle:
    |   - initialize
    |   - initialize_loggers
    |
    |   training loop:
    |       - update_ready
    |           - scheduled_update
    |               - update
    |       - scheduled_log_update
    |           - log_update
    |       - loss_update
    |       - optimizer_pre_step
    |       - optimizer_post_step

    :param log_types: The loggers that can be used by the modifier instance
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param log_types: The loggers that can be used by the modifier instance
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be less than, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        log_types: Union[str, List[str]] = None,
        start_epoch: float = -1.0,
        min_start: float = -1.0,
        end_epoch: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        **kwargs,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            min_start=min_start,
            end_epoch=end_epoch,
            min_end=min_end,
            end_comparator=end_comparator,
            **kwargs,
        )

        self._started = False
        self._ended = False
        self._schedule_called = False
        self._scheduled_log_called = False

        self.validate_schedule()

    @ModifierProp(serializable=False)
    def started(self) -> bool:
        """
        :return: True if the modifier has been started
            (ie between the start and end range), False otherwise
        """
        return self._started

    @ModifierProp(serializable=False)
    def ended(self) -> bool:
        """
        :return: True if the modifier has ended (ie after the start and end range),
            False otherwise
        """
        return self._ended

    def start_pending(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation compares current epoch with the start epoch.

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is ready to begin modifying, false otherwise
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        pending = (
            not self._started
            and not self._ended
            and (epoch >= self._start_epoch >= 0.0 or self._start_epoch == -1.0)
        )

        return pending

    def end_pending(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation compares current epoch with the end epoch and
        that it has been started.

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is ready to stop modifying, false otherwise
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        pending = not self._ended and self._started and epoch >= self._end_epoch >= 0.0

        return pending

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation checks if start_pending() or end_pending().

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        pending = self.start_pending(epoch, steps_per_epoch) or self.end_pending(
            epoch, steps_per_epoch
        )

        return pending

    def scheduled_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Called by the system and calls into update() method
        Tracks state and should not be overridden!!

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self.update_ready(epoch, steps_per_epoch):
            raise RuntimeError(
                "update_ready returns False, this must be true to call scheduled_update"
            )

        self._schedule_called = True
        self.update(module, optimizer, epoch, steps_per_epoch)
        self._schedule_called = False

        if self.start_pending(epoch, steps_per_epoch):
            self._started = True

        if self.end_pending(epoch, steps_per_epoch):
            self._ended = True

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles updating the modifier's state, module, or optimizer.
        Called when update_ready() returns True.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._schedule_called:
            raise RuntimeError(
                "update should not be called directly, call scheduled_update instead"
            )

    def scheduled_log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles checking if a log update should happen.
        IE, is the modifier currently in the range of its start and end epochs.
        No restrictions are placed on it by update_ready in the event that the modifier
        should log constantly or outside of an update being ready.
        General use case is checking if logs should happen by comparing
        cached values with updated values.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._loggers_initialized:
            raise RuntimeError("modifier must have loggers initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

        self._scheduled_log_called = True
        self.log_update(module, optimizer, epoch, steps_per_epoch)
        self._scheduled_log_called = False

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles logging updates for the modifier for better tracking and visualization.
        Should be overridden for logging but not called directly,
        use scheduled_log_update instead.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if not self._scheduled_log_called:
            raise RuntimeError(
                "log_update should not be called directly, "
                "call scheduled_log_update instead"
            )


class ScheduledUpdateModifier(ScheduledModifier, BaseUpdate):
    """
    The base scheduled update modifier implementation,
    all scheduled update modifiers must inherit from this class.
    The difference for this and a ScheduledModifier is that these have a certain
    interval that they update within the start and end ranges.
    It defines common things needed for the lifecycle and implementation of a scheduled
    update modifier.

    | Lifecycle:
    |   - initialize
    |   - initialize_loggers
    |
    |   training loop:
    |       - update_ready
    |           - scheduled_update
    |               - update
    |       - loss_update
    |       - optimizer_pre_step
    |       - optimizer_post_step

    :param log_types: The loggers that can be used by the modifier instance
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param log_types: The loggers that can be used by the modifier instance
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be less than, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param min_frequency: The minimum acceptable value for update_frequency, default -1
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        log_types: Union[str, List[str]] = None,
        start_epoch: float = -1.0,
        min_start: float = -1.0,
        end_epoch: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        update_frequency: float = -1.0,
        min_frequency: float = -1.0,
        **kwargs,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            min_start=min_start,
            end_epoch=end_epoch,
            min_end=min_end,
            end_comparator=end_comparator,
            update_frequency=update_frequency,
            min_frequency=min_frequency,
            **kwargs,
        )
        self._last_update_epoch = -1.0

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Calls base implementation to check if start_pending() or end_pending().
        Additionally checks if an update is ready based on the frequency and current'
        epoch vs last epoch updated.

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._enabled:
            return False

        start_or_end = super().update_ready(epoch, steps_per_epoch)
        update_ready = (
            self.started
            and epoch > self._start_epoch
            and not self.ended
            and (
                (self._update_frequency == -1.0)
                or (
                    self._last_update_epoch >= 0.0
                    and epoch >= self._last_update_epoch + self._update_frequency
                )
            )
        )

        return start_or_end or update_ready

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles updating the modifier's state, module, or optimizer.
        Called when update_ready() returns True.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        self._last_update_epoch = epoch

        return super().update(module, optimizer, epoch, steps_per_epoch)
