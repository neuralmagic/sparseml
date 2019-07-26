from abc import ABC, abstractmethod
from typing import List
import yaml
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer


__all__ = ['ALL_TOKEN', 'ScheduledModifier', 'ScheduledUpdateModifier', 'ScheduledModifierManager']


ALL_TOKEN = '__all__'


class Modifier(ABC):
    @abstractmethod
    def initialize(self, module: Module, optimizer: Optimizer):
        raise NotImplementedError()

    @abstractmethod
    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def loss_update(self, loss: Tensor, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def optimizer_pre_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def optimizer_post_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        raise NotImplementedError()


class ScheduledModifier(Modifier):
    def __init__(self, start_epoch: float = -1.0, end_epoch: float = -1.0):
        """
        Base class for any modifier
        Defines methods expected to be available for the rest of the system

        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        """
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._started = False
        self._ended = False

    @property
    def start_epoch(self) -> float:
        return self._start_epoch

    @property
    def end_epoch(self) -> float:
        return self._end_epoch

    @property
    def started(self) -> bool:
        return self._started

    @property
    def ended(self) -> bool:
        return self._ended

    def start_pending(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation compares current epoch with the start epoch

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is ready to begin modifying, false otherwise
        """
        pending = not self._started and not self._ended and (epoch >= self._start_epoch >= 0.0 or
                                                             self._start_epoch == -1.0)

        return pending

    def end_pending(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation compares current epoch with the end epoch and that it has been started

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is ready to stop modifying, false otherwise
        """
        pending = not self._ended and self._started and epoch >= self._end_epoch >= 0.0

        return pending

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Base implementation checks if start_pending() or end_pending()

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        pending = self.start_pending(epoch, steps_per_epoch) or self.end_pending(epoch, steps_per_epoch)

        return pending

    def scheduled_update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Called by the system and calls into update() method
        Tracks state and should not be overridden!!

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        self.update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            self._started = True

        if self.end_pending(epoch, steps_per_epoch):
            self._ended = True

    @abstractmethod
    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Handles initializing and setting up the modifier
        Called once on construction of the scheduled optimizer

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Handles updating the modifier's state, module, or optimizer
        Called when update_ready() returns True

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        raise NotImplementedError()

    def loss_update(self, loss: Tensor, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Optional call that can be made on the optimizer to update the modifiers once the loss has been calculated
        Called independent of if the modifier is currently active or not

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        pass

    def optimizer_pre_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Called before the optimizer step happens (after backward has been called, before optimizer.step)
        Called independent of if the modifier is currently active or not

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        pass

    def optimizer_post_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Called after the optimizer step happens and weights have updated
        Called independent of if the modifier is currently active or not

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        pass


class ScheduledUpdateModifier(ScheduledModifier):
    def __init__(self, start_epoch: float = -1.0, end_epoch: float = -1.0, update_frequency: float = -1.0):
        """
        Base class for any update modifier, allows updates to happen every # epochs (or fraction of epochs)
        Overrides update_ready to return true when update_frequency is reached

        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        """
        super().__init__(start_epoch, end_epoch)
        self._update_frequency = update_frequency
        self._last_update_epoch = -1.0

    @property
    def update_frequency(self) -> float:
        return self._update_frequency

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        Calls base implementation to check if start_pending() or end_pending()
        Additionally checks if an update is ready based on the frequency and current epoch vs last epoch updated

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        start_or_end = super().update_ready(epoch, steps_per_epoch)
        update_ready = (self.started and not self.ended and self._update_frequency >= 0.0 and
                        self._last_update_epoch >= 0.0 and epoch >= self._last_update_epoch + self._update_frequency)

        if start_or_end or update_ready:
            self._last_update_epoch = epoch

            return True

        return False

    @abstractmethod
    def initialize(self, module: Module, optimizer: Optimizer):
        raise NotImplementedError()

    @abstractmethod
    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        raise NotImplementedError()


class ScheduledModifierManager(Modifier):
    def __init__(self, modifiers: List[ScheduledModifier]):
        """
        Convenience wrapper around multiple scheduled modifiers

        :param modifiers: the modifiers to wrap
        """
        self._modifiers = modifiers

    def __del__(self):
        self._modifiers.clear()

    @property
    def min_epochs(self) -> int:
        vals = []
        vals.extend([mod.start_epoch for mod in self._modifiers if mod.start_epoch > -1])
        vals.extend([mod.end_epoch for mod in self._modifiers if mod.end_epoch > -1])

        return min(vals) if len(vals) > 0 else -1

    @property
    def max_epochs(self) -> int:
        vals = []
        vals.extend([mod.start_epoch for mod in self._modifiers if mod.start_epoch > -1])
        vals.extend([mod.end_epoch for mod in self._modifiers if mod.end_epoch > -1])

        return min(vals) if len(vals) > 0 else -1

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Handles initializing and setting up the contained modifiers
        Called once on construction of the scheduled optimizer

        :param module: module to modify
        :param optimizer: optimizer to modify
        """

        for mod in self._modifiers:
            mod.initialize(module, optimizer)

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Handles updating the contained modifiers' states, module, or optimizer
        Only calls scheduled_update on the each modifier if modifier.update_ready()

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        for mod in self._modifiers:
            if mod.update_ready(epoch, steps_per_epoch):
                mod.scheduled_update(module, optimizer, epoch, steps_per_epoch)

    def loss_update(self, loss: Tensor, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Optional call that can be made on the optimizer to update the contained modifiers once loss has been calculated

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """

        for mod in self._modifiers:
            mod.loss_update(loss, module, optimizer, epoch, steps_per_epoch)

    def optimizer_pre_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Called before the optimizer step happens (after backward has been called, before optimizer.step)
        Calls into the contained modifiers

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """

        for mod in self._modifiers:
            mod.optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)

    def optimizer_post_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Called after the optimizer step happens and weights have updated
        Calls into the contained modifiers

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """

        for mod in self._modifiers:
            mod.optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

    @staticmethod
    def from_yaml(file_path: str):
        """
        Convenience function used to create the manager of multiple modifiers from a yaml file

        :param file_path: the path to the yaml file to load the modifier from
        :return: ScheduledModifierManager() created from the yaml file
        """
        with open(file_path, 'r') as yaml_file:
            loaded = yaml.safe_load(yaml_file)
        manager = ScheduledModifierManager(loaded['modifiers'])

        return manager
