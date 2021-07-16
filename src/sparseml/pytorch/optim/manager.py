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

from typing import Any, Dict, List, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseManager
from sparseml.pytorch.optim.modifier import Modifier, ScheduledModifier
from sparseml.pytorch.utils import BaseLogger
from sparseml.utils import load_recipe_yaml_str
from sparsezoo.objects import Recipe


__all__ = ["RecipeManagerStepWrapper", "ScheduledModifierManager"]


class RecipeManagerStepWrapper(object):
    """
    A wrapper class to handle wrapping an optimizer or optimizer like object
    and override the step function.
    The override calls into the ScheduledModifierManager when appropriate and enabled
    and then calls step() as usual on the function with the original arguments.
    All original attributes and methods are forwarded to the wrapped object
    so this class can be a direct substitute for it.

    :param wrap: The object to wrap the step function and properties for.
    :param optimizer: The optimizer used in the training process.
    :param module: The model/module used in the training process.
    :param manager: The manager to forward lifecycle calls into such as step.
    :param epoch: The epoch to start the modifying process at.
    :param steps_per_epoch: The number of optimizer steps (batches) in each epoch.
    """

    def __init__(
        self,
        wrap: Any,
        optimizer: Optimizer,
        module: Module,
        manager: Any,
        epoch: float,
        steps_per_epoch: int,
    ):
        if not isinstance(manager, ScheduledModifierManager):
            raise ValueError(
                "manager must be an instance of a ScheduledModifierManager"
            )

        if not hasattr(wrap, "step"):
            raise ValueError("wrapped must have a step function")

        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be >= 0")

        # prefix everything with wrapped to avoid wrapped objects namespace
        self._wrapped = wrap
        self._wrapped_manager = manager
        self._wrapped_optimizer = optimizer
        self._wrapped_module = module
        self._wrapped_steps_per_epoch = steps_per_epoch
        self._wrapped_epoch = epoch
        self._wrapped_steps = round(epoch * steps_per_epoch)

    def __del__(self):
        try:
            # propagate delete to manager so it can disconnect hooks and objects
            del self._wrapped_manager
        except Exception:
            pass

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)

        return getattr(self._wrapped, item)

    def __setattr__(self, key, value):
        if key.startswith("_wrapped"):
            super().__setattr__(key, value)
        else:
            setattr(self._wrapped, key, value)

    @property
    def wrapped(self):
        """
        :return: The object to wrap the step function and properties for.
        """
        return self._wrapped

    @property
    def wrapped_manager(self):
        """
        :return: The manager to forward lifecycle calls into such as step.
        """
        return self._wrapped_manager

    @property
    def wrapped_optimizer(self) -> Any:
        """
        :return: The optimizer used in the training process.
        """
        return self._wrapped_optimizer

    @property
    def wrapped_module(self) -> Module:
        """
        :return: The model/module used in the training process.
        """
        return self._wrapped_module

    @property
    def wrapped_steps_per_epoch(self) -> int:
        """
        :return: The number of optimizer steps (batches) in each epoch.
        """
        return self._wrapped_steps_per_epoch

    @property
    def wrapped_epoch(self) -> float:
        """
        :return: The current epoch the wrapped object is at.
        """
        return self._wrapped_epoch

    @property
    def wrapped_steps(self) -> int:
        """
        :return: The current number of steps that have been called for
            the wrapped object.
        """
        return self._wrapped_steps

    def step(self, *args, **kwargs):
        """
        Override for the step function.
        Calls into the base step function with the args and kwargs.

        :param args: Any args to pass to the wrapped objects step function.
        :param kwargs: Any kwargs to pass to the wrapped objects step function.
        :return: The return, if any, from the wrapped objects step function
        """
        return self._perform_wrapped_step(*args, **kwargs)

    def emulated_step(self):
        """
        Emulated step function to be called in place of step when the
        number of steps_per_epoch vary across epochs.
        The emulated function should be called to keep the steps_per_epoch thee same.
        Does not call into the step function for the wrapped object,
        but does call into the manager to increment the steps.
        """
        self._perform_wrapped_step(skip_orig_step=True)

    def loss_update(self, loss: Tensor) -> Tensor:
        """
        Optional call to update modifiers based on the calculated loss.
        Not needed unless one or more of the modifier is using the loss
        to make a modification or is modifying the loss itself.

        :param loss: the calculated loss after running a forward pass and loss_fn
        :return: the modified loss tensor
        """
        loss = self._wrapped_manager.loss_update(
            loss,
            self._wrapped_module,
            self._wrapped_optimizer,
            self._wrapped_epoch,
            self._wrapped_steps_per_epoch,
        )

        return loss

    def _perform_wrapped_step(self, *args, **kwargs) -> Any:
        skip_orig_step = (
            kwargs["skip_orig_step"] if "skip_orig_step" in kwargs else False
        )
        ret = None

        if self._wrapped_manager.enabled:
            self._wrapped_manager.update(
                self._wrapped_module,
                self._wrapped_optimizer,
                self._wrapped_epoch,
                self._wrapped_steps_per_epoch,
            )
            self._wrapped_manager.optimizer_pre_step(
                self._wrapped_module,
                self._wrapped_optimizer,
                self._wrapped_epoch,
                self._wrapped_steps_per_epoch,
            )

        if not skip_orig_step:
            ret = self._wrapped.step(*args, **kwargs)

        if self._wrapped_manager.enabled:
            self._wrapped_manager.optimizer_post_step(
                self._wrapped_module,
                self._wrapped_optimizer,
                self._wrapped_epoch,
                self._wrapped_steps_per_epoch,
            )

        # update tracking metrics for epoch and steps
        self._wrapped_steps += 1
        epoch_num = self._wrapped_steps // self._wrapped_steps_per_epoch
        epoch_steps = self._wrapped_steps % self._wrapped_steps_per_epoch
        self._wrapped_epoch = float(epoch_num) + (
            float(epoch_steps) / float(self._wrapped_steps_per_epoch)
        )

        return ret


class ScheduledModifierManager(BaseManager, Modifier):
    """
    The base modifier manager, handles managing multiple ScheduledModifers.

    | Lifecycle:
    |   - initialize
    |   - initialize_loggers
    |   - modify
    |   - finalize

    :param modifiers: the modifiers to wrap
    """

    @staticmethod
    def from_yaml(
        file_path: Union[str, Recipe],
        add_modifiers: List[Modifier] = None,
    ):
        """
        Convenience function used to create the manager of multiple modifiers from a
        recipe file.

        :param file_path: the path to the recipe file to load the modifier from, or
            a SparseZoo model stub to load a recipe for a model stored in SparseZoo.
            SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
            '?recipe_type=<type>' parameter. Can also be a SparseZoo Recipe
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

    def __init__(self, modifiers: List[ScheduledModifier]):
        super().__init__(modifiers=modifiers)
        self._initialize_epoch = 0

    def state_dict(self) -> Dict[str, Dict]:
        """
        :return: Dictionary to store any state variables for this manager.
            Includes all modifiers nested under this manager as sub keys in the dict.
            Only modifiers that a non empty state dict are included.
        """
        state_dict = {mod.identifier(): mod.state_dict() for mod in self.modifiers}

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Dict], strict: bool = True):
        """
        Loads the given state dict into this manager.
        All modifiers that match will be loaded.
        If any are missing or extra and strict=True, then will raise a KeyError

        :param state_dict: dictionary object as generated by this object's state_dict
            function
        :param strict: True to raise a KeyError for any missing or extra information in
            the state dict, False to ignore
        :raises IndexError: If any keys in the state dict do not correspond to a valid
            index for this manager and strict=True
        """
        modifiers_index = {mod.identifier(): mod for mod in self.modifiers}

        if strict:
            modifier_keys = {key for key in modifiers_index.keys()}
            state_dict_keys = {key for key in state_dict.keys()}
            diff = modifier_keys.symmetric_difference(state_dict_keys)
            if diff:
                raise IndexError(
                    f"Found extra keys: {state_dict_keys - modifier_keys} "
                    f"and missing keys: {modifier_keys - state_dict_keys}"
                )

        for key, val in state_dict.items():
            if key not in modifiers_index:
                continue

            modifiers_index[key].load_state_dict(val)

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Handles any initialization of the manager for the given model/module.
        epoch and steps_per_epoch can optionally be passed in to initialize the manager
        and module at a specific point in the training process.
        If loggers is not None, will additionally call initialize_loggers.

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the manager and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)
        self._initialize_epoch = epoch

        for mod in self._modifiers:
            mod.initialize(module, epoch, loggers, **kwargs)

    def initialize_loggers(self, loggers: Union[None, List[BaseLogger]]):
        """
        Handles initializing and setting up the loggers for the contained modifiers.

        :param loggers: the loggers to setup this manager with for logging important
            info and milestones to
        """
        super().initialize_loggers(loggers)

        for mod in self._modifiers:
            mod.initialize_loggers(loggers)

    def modify(
        self,
        module: Module,
        optimizer: Optimizer,
        steps_per_epoch: int,
        wrap_optim: Any = None,
        epoch: float = None,
    ) -> RecipeManagerStepWrapper:
        """
        Modify the given module and optimizer for training aware algorithms such as
        pruning and quantization.
        Initialize must be called first.
        After training is complete, finalize should be called.

        :param module: The model/module to modify
        :param optimizer: The optimizer to modify
        :param steps_per_epoch: The number of optimizer steps (batches) in each epoch
        :param wrap_optim: Optional object to wrap instead of the optimizer.
            Useful for cases like amp (fp16 training) where a it should be wrapped
            in place of the original optimizer since it doesn't always call into
            the optimizer.step() function.
        :param epoch: Optional epoch that can be passed in to start modifying at.
            Defaults to the epoch that was supplied to the initialize function.
        :return: A wrapped optimizer object. The wrapped object makes all the
            original properties for the wrapped object available so it can be
            used without any additional code changes.
        """
        if epoch is None:
            epoch = self._initialize_epoch

        if not self.initialized:
            self.initialize(module, epoch)

        if wrap_optim is None:
            wrap_optim = optimizer

        return RecipeManagerStepWrapper(
            wrap_optim, optimizer, module, self, epoch, steps_per_epoch
        )

    def finalize(
        self, module: Optional[Module] = None, reset_loggers: bool = True, **kwargs
    ):
        """
        Handles any finalization of the modifier for the given model/module.
        Applies any remaining logic and cleans up any hooks or attachments to the model.

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().finalize(module, reset_loggers, **kwargs)

        for mod in self._modifiers:
            mod.finalize(module, reset_loggers, **kwargs)

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
