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
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

import weakref
from functools import wraps
from typing import Dict, List, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseManager
from sparseml.pytorch.optim.modifier import Modifier, ScheduledModifier
from sparseml.pytorch.utils import PyTorchLogger
from sparseml.utils import load_recipe_yaml_str
from sparsezoo.objects import OptimizationRecipe


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
    def from_yaml(
        file_path: Union[str, OptimizationRecipe],
        add_modifiers: List[Modifier] = None,
    ):
        """
        Convenience function used to create the manager of multiple modifiers from a
        recipe file.

        :param file_path: the path to the recipe file to load the modifier from, or
            a SparseZoo model stub to load a recipe for a model stored in SparseZoo.
            SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
            '?recipe_type=<type>' parameter. Can also be a SparseZoo OptimizationRecipe
            object. i.e. '/path/to/local/recipe.yaml', 'zoo:model/stub/path',
            'zoo:model/stub/path?recipe_type=transfer'
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the recipe file
        :return: ScheduledModifierManager() created from the recipe file
        """
        yaml_str = load_recipe_yaml_str(file_path)
        modifiers = Modifier.load_list(yaml_str)

        if add_modifiers:
            modifiers.extend(add_modifiers)

        manager = ScheduledModifierManager(modifiers)

        return manager

    @staticmethod
    def adjust_optimizer_step(optimizer: Optimizer, epoch: int, step: int):
        """
        Adjust the current step for the optimizer's managed schedule to the given
        epoch and step.

        :param optimizer: the manager initialized optimizer to adjust the step for
        :param epoch: the epoch to set the current global step to match
        :param step: the step (batch) within the epoch to set the
            current global step to match
        """
        if not getattr(optimizer.step, "_with_modifiers", False):
            raise RuntimeError(
                "Optimizer not initialized with ScheduledModifierManager.initialize"
            )
        optimizer._steps = epoch * optimizer._steps_per_epoch + step
        _set_scheduled_epoch(optimizer)

    def __init__(self, modifiers: List[ScheduledModifier]):
        super().__init__(modifiers=modifiers)

    def initialize(
        self,
        module: Module,
        optimizer: Optimizer,
        steps_per_epoch: int,
        loggers: Union[List[PyTorchLogger], None] = None,
    ):
        """
        Handles initializing and setting up the contained modifiers
        Called once on construction of the scheduled optimizer

        :param optimizer: optimizer to modify
        :param module: module to modify
        :param steps_per_epoch: the number of steps or batches in each epoch,
            used to calculate decimals within the epoch
        :param loggers: loggers to log important info to within the modifiers;
            ex tensorboard or to the console
        """
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be >= 0")

        super().initialize(module, optimizer)

        for mod in self._modifiers:
            mod.initialize(module, optimizer)

        self._modify_optimizer_step(module, optimizer, steps_per_epoch)

        self.initialize_loggers(loggers)

    def _modify_optimizer_step(
        self, module: Module, optimizer: Optimizer, steps_per_epoch: int
    ):
        def _step_with_modifiers(step_method):
            if getattr(step_method, "_with_modifiers", False):
                # `optimizer.step()` has already been replaced, return.
                return step_method

            # Prevent cyclic references by keeping a weak reference
            # to optimizer class and original unbound step method
            optim_ref = weakref.ref(step_method.__self__)
            original_step_func = step_method.__func__
            optim_cls = optim_ref().__class__
            del step_method

            recipe_manager = self

            @wraps(original_step_func)
            def modifier_step_wrapper(*args, **kwargs):
                optim_instance = optim_ref()

                # set current epoch
                _set_scheduled_epoch(optim_instance)

                # run modifiers
                recipe_manager.update(
                    module,
                    optim_instance,
                    optim_instance._epoch,
                    optim_instance._steps_per_epoch,
                )
                recipe_manager.optimizer_pre_step(
                    module,
                    optim_instance,
                    optim_instance._epoch,
                    optim_instance._steps_per_epoch,
                )

                # optimizer step
                optim_outputs = original_step_func.__get__(optim_instance, optim_cls)(
                    *args, **kwargs
                )

                # post step hooks
                recipe_manager.optimizer_post_step(
                    module,
                    optim_instance,
                    optim_instance._epoch,
                    optim_instance._steps_per_epoch,
                )
                optim_instance._steps += 1

                return optim_outputs

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            modifier_step_wrapper._with_modifiers = True
            modifier_step_wrapper._original_step_func = original_step_func
            return modifier_step_wrapper

        # wrap optimizer step method
        optimizer.step = _step_with_modifiers(optimizer.step)
        optimizer._epoch = 0
        optimizer._steps = 0
        optimizer._steps_per_epoch = steps_per_epoch

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

    def finalize(self, module: Module, optimizer: Optimizer):
        """
        Remove extra information and hooks added to the module and optimizer
        by the Modifier.

        :param module: module to finalize
        :param optimizer: optimizer to finalize
        """
        super().finalize(module, optimizer)
        for mod in self._modifiers:
            mod.finalize(module, optimizer)

        # revert optimizer to use original step, do not invoke manager
        original_step_func = getattr(optimizer.step, "_original_step_func", None)
        if original_step_func:
            # delete wrapped step function and added variables
            del optimizer.step  # delete wrapped step function
            del optimizer._epoch
            del optimizer._steps
            del optimizer._steps_per_epoch

            # bind unbound original step function back to optimizer instance and reset
            bound_original_step_func = original_step_func.__get__(
                optimizer, optimizer.__class__
            )
            setattr(optimizer, "step", bound_original_step_func)


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


def _set_scheduled_epoch(optimizer: Optimizer):
    epoch_num = optimizer._steps // optimizer._steps_per_epoch
    epoch_steps = optimizer._steps % optimizer._steps_per_epoch
    optimizer._epoch = float(epoch_num) + float(epoch_steps) / float(
        optimizer._steps_per_epoch
    )
