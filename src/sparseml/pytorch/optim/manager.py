"""
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

from typing import Dict, List, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseManager
from sparseml.pytorch.optim.modifier import Modifier, ScheduledModifier
from sparseml.pytorch.utils import PyTorchLogger
from sparseml.utils import load_recipe_yaml_str


__all__ = ["ScheduledModifierManager", "load_manager"]


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
        yaml_str = load_recipe_yaml_str(file_path)
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

    def state_dict(self) -> Dict[str, Dict]:
        """
        :return: Dictionary to store any state variables from this Manager's Modifiers.
            Only Modifiers with a state_dict function will be included. The mapping
            is modifier_idx -> modifier.state_dict(). If no modifiers have a state
            dict, an empty dictionary is returned.
        """
        return {
            str(idx): modifier.state_dict()
            for idx, modifier in enumerate(self.modifiers)
            if hasattr(modifier, "state_dict")
        }

    def load_state_dict(self, state_dict: Dict[str, Dict]):
        """
        Loads the given state dict into this object's modifiers

        :param state_dict: dictionary object as generated by this object's state_dict
            function
        :raises RuntimeError: If any keys in the state dict do not correspond to a valid
            index in this manager's modifier list
        """
        num_moidifiers = len(self.modifiers)
        modifier_keys = [int(idx) for idx in state_dict.keys()]
        if any(idx < 0 or idx >= num_moidifiers for idx in modifier_keys):
            raise RuntimeError(
                "Invalid modifier index in state dict for ScheduledModifierManager for"
                "ScheduledModifierManager with {} modifiers. Given indices: {}".format(
                    num_moidifiers, modifier_keys
                )
            )
        for idx, modifier_state_dict in state_dict.items():
            self.modifiers[int(idx)].load_state_dict(modifier_state_dict)

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


def load_manager(
    path: str, manager: ScheduledModifierManager, map_location: Union[None, str] = "cpu"
):
    """
    Load the state dict into a ScheduledModifierManager from a given file.

    :param path: the path to the pth file to load the state dict from
    :param manager: the optimizer to load the state dict into
    :param map_location: the location to map the values to when loading
    """
    state_dict = torch.load(path, map_location=map_location)
    if "manager" in state_dict:
        state_dict = state_dict["manager"]
    manager.load_state_dict(state_dict)
