"""
Contains base code related to modifiers: objects that modify some aspect of the training process for a model
For example, learning rate schedules or kernel sparsity (weight pruning) are implemented as modifiers
"""


from abc import ABC
from typing import Union, List
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from ..utils import ALL_TOKEN, validate_str_list
from .logger import ModifierLogger


__all__ = [
    "Modifier",
    "ScheduledModifier",
    "ScheduledUpdateModifier",
]


class Modifier(ABC):
    """
    The base modifier implementation, all modifiers must inherit from this class.
    It defines common things needed for the lifecycle and implementation of a modifier.

    Lifecycle:
        - initialize
        - initialize_loggers

        training loop:
            - update
            - log_update
            - loss_update
            - optimizer_pre_step
            - optimizer_post_step
    """

    def __init__(self, allowed_loggers: Union[None, str, List[str]] = ALL_TOKEN):
        """
        :param allowed_loggers: the names of the loggers the modifier can log to. set to ALL_TOKEN for no restriction
                                and None for no loggers
        """
        self._allowed_loggers = (
            validate_str_list(
                allowed_loggers, "logger_names for {}".format(self.__class__.__name__)
            )
            if allowed_loggers
            else None
        )
        self._initialized = False
        self._loggers_initialized = False
        self._loggers = None
        self._enabled = True

    @property
    def allowed_loggers(self) -> Union[None, str, List[str]]:
        """
        :return: the names of the loggers the modifier can log to. set to ALL_TOKEN for no restriction
                 and None for no loggers
        """
        return self._allowed_loggers

    @property
    def initialized(self) -> bool:
        """
        :return: True if the modifier has gone through the initialized life cycle, False otherwise
        """
        return self._initialized

    @property
    def loggers_initialized(self):
        return self._loggers_initialized

    @property
    def loggers(self):
        """
        :return: loggers to log important info to within for this modifier (filtered by allowed_loggers)
        """
        return self._loggers if self._loggers is not None else []

    @property
    def enabled(self) -> bool:
        """
        :return: True if the modifier is currently enabled and making updates, False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to allow the modifier to make updates, False otherwise
        """
        self._enabled = value

    def prop_set_check(self, prop_name: str = ""):
        """
        :param prop_name: name to raise the error under
        :return: checks that the given properties won't be set if modifier has already been initialized
        """
        if self._initialized or self._loggers_initialized:
            raise RuntimeError(
                "Cannot change {} after {} has been initialized".format(
                    prop_name, self.__class__.__name__
                )
            )

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Handles initializing and setting up the modifier
        Called once on construction of the scheduled optimizer

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        self._initialized = True

    def initialize_loggers(self, loggers: Union[None, List[ModifierLogger]]):
        """
        :param loggers: the loggers to setup this modifier with for logging important info and milestones to
        """
        self._loggers_initialized = True

        if not self._allowed_loggers or not loggers:
            return

        self._loggers = [
            log
            for log in loggers
            if self._allowed_loggers == ALL_TOKEN or log.name in self._allowed_loggers
        ]

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles updating the modifier's state, module, or optimizer
        Called when update_ready() returns True

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles logging updates for the modifier for better tracking and visualization
        Should be overriden for logging

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
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
        Optional call that can be made on the optimizer to update the modifiers once the loss has been calculated
        Called independent of if the modifier is currently active or not

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
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
        Called before the optimizer step happens (after backward has been called, before optimizer.step)
        Called independent of if the modifier is currently active or not

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Called after the optimizer step happens and weights have updated
        Called independent of if the modifier is currently active or not

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")


class ScheduledModifier(Modifier):
    """
    The base scheduled modifier implementation, all scheduled modifiers must inherit from this class.
    The difference for this and a Modifier is that these have start and end epochs.
    It defines common things needed for the lifecycle and implementation of a scheduled modifier.

    Lifecycle:
        - initialize
        - initialize_loggers

        training loop:
            - update_ready
                - scheduled_update
                    - update
            - scheduled_log_update
                - log_update
            - loss_update
            - optimizer_pre_step
            - optimizer_post_step
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        allowed_loggers: Union[None, str, List[str]] = ALL_TOKEN,
    ):
        """
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        :param allowed_loggers the names of the loggers the modifier can log to. set to ALL_TOKEN for no restriction
                                and None for no loggers
        """
        super(ScheduledModifier, self).__init__(allowed_loggers)
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._started = False
        self._ended = False
        self._schedule_called = False
        self._scheduled_log_called = False

    @property
    def start_epoch(self) -> float:
        """
        :return: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        """
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value: float):
        """
        :param value: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change start_epoch after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._start_epoch = value

    @property
    def end_epoch(self) -> float:
        """
        :return: The epoch to end the modifier at (set to -1.0 so it never ends)
        """
        return self._end_epoch

    @end_epoch.setter
    def end_epoch(self, value: float):
        """
        :param value: The epoch to end the modifier at (set to -1.0 so it never ends)
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change end_epoch after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._end_epoch = value

    @property
    def started(self) -> bool:
        """
        :return: True if the modifier has been started (ie between the start and end range), False otherwise
        """
        return self._started

    @property
    def ended(self) -> bool:
        """
        :return: True if the modifier has ended (ie after the start and end range), False otherwise
        """
        return self._ended

    def start_pending(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation compares current epoch with the start epoch

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is ready to begin modifying, false otherwise
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self.enabled:
            return False

        pending = (
            not self._started
            and not self._ended
            and (epoch >= self._start_epoch >= 0.0 or self._start_epoch == -1.0)
        )

        return pending

    def end_pending(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation compares current epoch with the end epoch and that it has been started

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is ready to stop modifying, false otherwise
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self.enabled:
            return False

        pending = not self._ended and self._started and epoch >= self._end_epoch >= 0.0

        return pending

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation checks if start_pending() or end_pending()

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self.enabled:
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
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
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
        Handles updating the modifier's state, module, or optimizer
        Called when update_ready() returns True

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if not self._schedule_called:
            raise RuntimeError(
                "update should not be called directly, call scheduled_update instead"
            )

    def scheduled_log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles checking if a log update should happen
        IE, is the modifier currently in the range of its start and end epochs
        No restrictions are placed on it by update_ready in the event that the modifier should log
        constantly or outside of an update being ready
        General use case is checking if logs should happen by comparing cached values with updated values

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._loggers_initialized:
            raise RuntimeError("modifier must have loggers initialized first")

        if not self._enabled:
            raise RuntimeError("modifier must be enabled")

        if epoch < self.start_epoch or epoch > self.end_epoch:
            return

        self._scheduled_log_called = True
        self.log_update(module, optimizer, epoch, steps_per_epoch)
        self._scheduled_log_called = False

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Handles logging updates for the modifier for better tracking and visualization
        Should be overridden for logging but not called directly, use scheduled_log_update instead

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if not self._scheduled_log_called:
            raise RuntimeError(
                "log_update should not be called directly, call scheduled_log_update instead"
            )


class ScheduledUpdateModifier(ScheduledModifier):
    """
    The base scheduled update modifier implementation, all scheduled update modifiers must inherit from this class.
    The difference for this and a ScheduledModifier is that these have a certain interval that they update
    within the start and end ranges.
    It defines common things needed for the lifecycle and implementation of a scheduled update modifier.

    Lifecycle:
        - initialize
        - initialize_loggers

        training loop:
            - update_ready
                - scheduled_update
                    - update
            - loss_update
            - optimizer_pre_step
            - optimizer_post_step
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        allowed_loggers: Union[None, str, List[str]] = ALL_TOKEN,
    ):
        """
        Base class for any update modifier, allows updates to happen every # epochs (or fraction of epochs)
        Overrides update_ready to return true when update_frequency is reached

        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        :param allowed_loggers the names of the loggers the modifier can log to. set to ALL_TOKEN for no restriction
                                and None for no loggers
        """
        super().__init__(start_epoch, end_epoch, allowed_loggers)
        self._update_frequency = update_frequency
        self._last_update_epoch = -1.0

    @property
    def update_frequency(self) -> float:
        """
        :return: The number of epochs or fraction of epochs to update at between start and end
        """
        return self._update_frequency

    @update_frequency.setter
    def update_frequency(self, value: float):
        """
        :param value: The number of epochs or fraction of epochs to update at between start and end
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change update_frequency after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._update_frequency = value

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Calls base implementation to check if start_pending() or end_pending()
        Additionally checks if an update is ready based on the frequency and current epoch vs last epoch updated

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self.enabled:
            return False

        start_or_end = super().update_ready(epoch, steps_per_epoch)
        update_ready = (
            self.started
            and epoch > self.start_epoch
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
        Handles updating the modifier's state, module, or optimizer
        Called when update_ready() returns True

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        self._last_update_epoch = epoch

        return super().update(module, optimizer, epoch, steps_per_epoch)
