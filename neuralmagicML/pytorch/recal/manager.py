"""
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

from typing import List, Union
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from neuralmagicML.recal import BaseManager
from neuralmagicML.utils import create_parent_dirs, clean_path
from neuralmagicML.pytorch.utils import PyTorchLogger
from neuralmagicML.pytorch.recal.modifier import Modifier, ScheduledModifier


__all__ = ["ScheduledModifierManager"]


class ScheduledModifierManager(BaseManager, Modifier):
    """
    The base modifier manager, handles managing multiple ScheduledModifers.

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

    :param modifiers: the modifiers to wrap
    """

    @staticmethod
    def from_yaml(file_path: str, add_modifiers: List[Modifier] = None):
        """
        Convenience function used to create the manager of multiple modifiers from a
        yaml file.

        :param file_path: the path to the yaml file to load the modifier from
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the yaml file
        :return: ScheduledModifierManager() created from the yaml file
        """
        with open(file_path, "r") as yaml_file:
            yaml_str = yaml_file.read()

        modifiers = Modifier.load_list(yaml_str)

        if add_modifiers:
            modifiers.extend(add_modifiers)

        manager = ScheduledModifierManager(modifiers)

        return manager

    def __init__(self, modifiers: List[ScheduledModifier]):
        super().__init__(modifiers=modifiers)

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Handles initializing and setting up the contained modifiers
        Called once on construction of the scheduled optimizer

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super().initialize(module, optimizer)

        for mod in self._modifiers:
            mod.initialize(module, optimizer)

    def initialize_loggers(self, loggers: Union[None, List[PyTorchLogger]]):
        """
        Handles initializing and setting up the loggers for the contained modifiers
        Called once on construction of the scheduled optimizer

        :param loggers: the loggers to setup this modifier with for logging important
            info and milestones to
        """
        super().initialize_loggers(loggers)

        for mod in self._modifiers:
            mod.initialize_loggers(loggers)

    def update(
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
        log_updates: bool = True,
    ):
        """
        Handles updating the contained modifiers' states, module, or optimizer
        Only calls scheduled_update on the each modifier if modifier.update_ready()

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :param log_updates: True to log the updates for each modifier to the loggers,
            False to skip logging
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        for mod in self._modifiers:
            if not mod.enabled:
                continue

            if mod.update_ready(epoch, steps_per_epoch):
                mod.scheduled_update(module, optimizer, epoch, steps_per_epoch)

            if log_updates:
                mod.scheduled_log_update(module, optimizer, epoch, steps_per_epoch)

    def loss_update(
        self,
        loss: Tensor,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ) -> Tensor:
        """
        Optional call that can be made on the optimizer to update the contained
        modifiers once loss has been calculated

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: the modified loss tensor
        """
        super().loss_update(loss, module, optimizer, epoch, steps_per_epoch)

        for mod in self._modifiers:
            if not mod.enabled:
                continue

            loss = mod.loss_update(loss, module, optimizer, epoch, steps_per_epoch)

        return loss

    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Called before the optimizer step happens (after backward has been called,
        before optimizer.step)
        Calls into the contained modifiers

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)

        for mod in self._modifiers:
            if not mod.enabled:
                continue

            mod.optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Called after the optimizer step happens and weights have updated
        Calls into the contained modifiers

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        for mod in self._modifiers:
            if not mod.enabled:
                continue

            mod.optimizer_post_step(module, optimizer, epoch, steps_per_epoch)
