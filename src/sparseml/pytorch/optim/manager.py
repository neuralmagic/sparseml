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

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import (
    BaseManager,
    add_framework_metadata,
    load_recipe_yaml_str,
    parse_recipe_variables,
    validate_metadata,
)
from sparseml.pytorch.sparsification.modifier import Modifier, ScheduledModifier
from sparseml.pytorch.utils import BaseLogger, LoggerManager, is_parallel_model
from sparsezoo import File


__all__ = ["RecipeManagerStepWrapper", "ScheduledModifierManager"]


_LOGGER = logging.getLogger(__name__)


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
        file_path: Union[str, File],
        add_modifiers: Optional[List[Modifier]] = None,
        recipe_variables: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Convenience function used to create the manager of multiple modifiers from a
        recipe file.

        :param file_path: the path to the recipe file to load the modifier from, or
            a SparseZoo model stub to load a recipe for a model stored in SparseZoo.
            SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
            '?recipe_type=<type>' parameter. Can also be a SparseZoo File
            object. i.e. '/path/to/local/recipe.md', 'zoo:model/stub/path',
            'zoo:model/stub/path?recipe_type=transfer'. Additionally, a raw
             yaml str is also supported in place of a file path.
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the recipe file
        :param recipe_variables: additional arguments to override any root variables
            in the recipe with (i.e. num_epochs, init_lr)
        :metadata: additional (to the information provided in the recipe) data to be
            preserved and utilized in the future - for reproducibility and completeness.
        :return: ScheduledModifierManager() created from the recipe file
        """
        recipe_variables = parse_recipe_variables(recipe_variables)
        yaml_str = load_recipe_yaml_str(file_path, **recipe_variables)
        modifiers = Modifier.load_list(yaml_str)
        if add_modifiers:
            modifiers.extend(add_modifiers)

        validated_metadata = validate_metadata(metadata, yaml_str)

        if metadata is not None:
            validated_metadata = add_framework_metadata(
                validated_metadata, torch_version=torch.__version__
            )

        manager = ScheduledModifierManager(
            modifiers=modifiers, metadata=validated_metadata
        )
        return manager

    def __init__(
        self,
        modifiers: List[ScheduledModifier],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(modifiers=modifiers, metadata=metadata)
        self._initialize_epoch = 0

    def state_dict(self) -> Dict[str, Dict]:
        """
        :return: Dictionary to store any state variables for this manager.
            Includes all modifiers nested under this manager as sub keys in the dict.
            Only modifiers that a non empty state dict are included.
        """

        def _modifiers_list_state_dict(modifiers):
            return {mod.identifier(): mod.state_dict() for mod in modifiers}

        if isinstance(self.modifiers, List):
            state_dict = _modifiers_list_state_dict(self.modifiers)
        else:
            state_dict = {
                stage: _modifiers_list_state_dict(modifiers)
                for stage, modifiers in self.modifiers
            }

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
        if isinstance(self.modifiers, List):
            modifiers_index = {mod.identifier(): mod for mod in self.modifiers}
        else:
            if strict:
                modifiers_stages = set(self.modifiers.keys())
                state_dict_stages = set(state_dict.keys())
                diff = modifiers_stages.symmetric_difference(state_dict_stages)
                if diff:
                    raise IndexError(
                        f"Found extra stages: {state_dict_stages - modifiers_stages}"
                        f"and missing stages: {modifiers_stages - state_dict_stages}"
                    )
            modifiers_index = {}
            for stage_modifiers in self.modifiers.values():
                modifiers_index.update(
                    {mod.identifier(): mod for mod in stage_modifiers}
                )

        if strict:
            modifier_keys = set(modifiers_index.keys())
            state_dict_keys = set(state_dict.keys())
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

    def apply(
        self,
        module: Module,
        epoch: float = math.inf,
        loggers: Optional[LoggerManager] = None,
        finalize: bool = True,
        **kwargs,
    ):
        """
        Applies the lifecycle of each stage in the manager/recipe
        by calling into initialize and finalize for each modifier for each stage

        :param module: the PyTorch model/module to modify
        :param epoch: the epoch to apply the modifier at, defaults to math.inf (end)
        :param loggers: Optional logger manager to log the modification process to
        :param finalize: True to invoke finalize after initialize, False otherwise.
            If training after one shot, set finalize=False to keep modifiers applied.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers (passed to initialize and finalize).
        """
        if not self.initialized:
            super().initialize(module, epoch, loggers, **kwargs)
            self._initialize_epoch = epoch

        modifier_lists = (
            self._modifiers
            if isinstance(self._modifiers, List)
            else list(self._modifiers.values())
        )

        for modifier_list in modifier_lists:

            self._initialize_modifiers(
                modifier_list, module, epoch, loggers=loggers, **kwargs
            )

            if finalize:
                self._finalize_modifiers(modifier_list, module, **kwargs)

    def apply_structure(
        self,
        module: Module,
        epoch: float = 0.0,
        loggers: Union[None, LoggerManager, List[BaseLogger]] = None,
        finalize: bool = False,
        **kwargs,
    ):
        """
        Initialize/apply the modifier for a given model/module at the given epoch
        if the modifier affects the structure of the module such as
        quantization, layer pruning, or filter pruning.
        Calls into initialize(module, epoch, loggers, **kwargs) if structured.

        :param module: the PyTorch model/module to modify
        :param epoch: the epoch to apply the modifier at, defaults to 0.0 (start)
        :param loggers: Optional logger manager to log the modification process to
        :param finalize: True to invoke finalize after initialize, False otherwise.
            Set finalize to True and epoch to math.inf for one shot application.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers (passed to initialize and finalize).
        """
        self._initialize_epoch = epoch
        for mod in self.iter_modifiers():
            mod.apply_structure(module, epoch, loggers, finalize, **kwargs)

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Union[None, LoggerManager, List[BaseLogger]] = None,
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
        :param loggers: Optional logger manager to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)
        self._initialize_epoch = epoch

        self._initialize_modifiers(
            self.iter_modifiers(), module, epoch, loggers, **kwargs
        )

    def initialize_loggers(self, loggers: Union[None, LoggerManager, List[BaseLogger]]):
        """
        Handles initializing and setting up the loggers for the contained modifiers.

        :param loggers: the logger manager to setup this manager with for logging
            important info and milestones to
        """
        super().initialize_loggers(loggers)

        for mod in self.iter_modifiers():
            mod.initialize_loggers(loggers)

    def modify(
        self,
        module: Module,
        optimizer: Optimizer,
        steps_per_epoch: int,
        wrap_optim: Any = None,
        epoch: float = None,
        allow_parallel_module: bool = True,
        **kwargs,
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
        :param allow_parallel_module: if False, a DataParallel or
            DistributedDataParallel module passed to this function will be unwrapped
            to its base module during recipe initialization by referencing
            module.module. This is useful so a recipe may reference the base module
            parameters instead of the wrapped distributed ones. Set to True to not
            unwrap the distributed module. Default is True
        :param kwargs: Key word arguments that are passed to the intialize call
            if initilaize has not been called yet
        :return: A wrapped optimizer object. The wrapped object makes all the
            original properties for the wrapped object available so it can be
            used without any additional code changes.
        """
        if epoch is None:
            epoch = self._initialize_epoch

        if is_parallel_model(module) and not allow_parallel_module:
            if allow_parallel_module:
                _LOGGER.warning(
                    "Parallel module detected by ScheduledModifierManager. Note that "
                    "the base module parameters will be prefixed by 'module.' which "
                    "may lead to matching issues if unaccounted for in recipe. Run "
                    "modify() with allow_parallel_module=False to unwrap the parallel "
                    "module during recipe initialization"
                )
            else:
                _LOGGER.info("Unwrapping parallel module for recipe initialization")
                module = module.module  # unwrap parallel module

        if not self.initialized:
            self.initialize(module, epoch, **kwargs)

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

        self._finalize_modifiers(self.iter_modifiers(), module, reset_loggers, **kwargs)

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

        for mod in self.iter_modifiers():
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
        **kwargs,
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
        super().loss_update(loss, module, optimizer, epoch, steps_per_epoch, **kwargs)

        for mod in self.iter_modifiers():
            if not mod.enabled:
                continue

            loss = mod.loss_update(
                loss,
                module,
                optimizer,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                **kwargs,
            )

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

        for mod in self.iter_modifiers():
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

        for mod in self.iter_modifiers():
            if not mod.enabled:
                continue

            mod.optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

    def _initialize_modifiers(
        self,
        modifiers: Iterable[Modifier],
        module: Module,
        epoch: float = 0,
        loggers: Union[None, LoggerManager, List[BaseLogger]] = None,
        **kwargs,
    ):
        if isinstance(modifiers, Modifier):
            modifiers = [modifiers]

        for mod in modifiers:
            if mod.initialized:
                # check in case modifier was initialized from apply_structure
                continue

            mod.initialize(module, epoch, loggers, **kwargs)

    def _finalize_modifiers(
        self,
        modifiers: Iterable[Modifier],
        module: Optional[Module] = None,
        reset_loggers: bool = True,
        **kwargs,
    ):
        if isinstance(modifiers, Modifier):
            modifiers = [modifiers]

        for mod in modifiers:
            mod.finalize(module, reset_loggers, **kwargs)
