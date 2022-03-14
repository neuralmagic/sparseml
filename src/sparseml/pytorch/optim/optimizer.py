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
Optimizer wrapper for enforcing Modifiers on the training process of a Module.
"""

import warnings
from typing import Any, Dict, List, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.optim.manager import (
    RecipeManagerStepWrapper,
    ScheduledModifierManager,
)
from sparseml.pytorch.utils import (
    BaseLogger,
    get_optim_learning_rate,
    set_optim_learning_rate,
)


__all__ = ["ScheduledOptimizer"]


class ScheduledOptimizer(Optimizer):
    """
    An optimizer wrapper to handle applying modifiers according to their schedule
    to both the passed in optimizer and the module.

    Overrides the step() function so that this method can call before and after on the
    modifiers to apply appropriate modifications to both the optimizer and the module.

    The epoch_start and epoch_end are based on how many steps have been taken
    along with the steps_per_epoch.

    | Lifecycle:
    |   - training cycle
    |       - zero_grad
    |       - loss_update
    |           - modifiers.loss_update
    |       - step
    |           - modifiers.update
    |           - modifiers.optimizer_pre_step
    |           - optimizer.step
    |           - modifiers.optimizers_post_step

    :param module: module to modify
    :param optimizer: optimizer to modify
    :param manager: the manager or list of managers used to apply modifications
    :param steps_per_epoch: the number of steps or batches in each epoch,
        not strictly required and can be set to -1.
        used to calculate decimals within the epoch,
        when not using can result in irregularities
    :param loggers: logger manager to log important info to within the modifiers;
        ex tensorboard or to the console
    :param initialize_kwargs: key word arguments and values to be passed to
        the recipe manager initialize function
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: Module,
        manager: ScheduledModifierManager,
        steps_per_epoch: int,
        loggers: Union[List[BaseLogger], None] = None,
        initialize_kwargs: Dict[str, Any] = None,
    ):
        # do not call into super since this instance is not passing all calls to
        # the nested optimizer
        # warnings.warn(
        #     "ScheduledOptimizer is deprecated and will be deleted in the future. "
        #     "Please replace with manager.modify",
        #     UserWarning,
        # )  TODO: uncomment in next release once docs are ready

        initialize_kwargs = initialize_kwargs or {}
        manager.initialize(module, epoch=0.0, loggers=loggers, **initialize_kwargs)
        self._wrapper = RecipeManagerStepWrapper(
            optimizer,
            optimizer,
            module,
            manager,
            epoch=0.0,
            steps_per_epoch=steps_per_epoch,
        )

    def __del__(self):
        try:
            del self._wrapper
        except Exception:
            pass

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)

        return getattr(self._wrapper.wrapped_optimizer, item)

    def __setattr__(self, key, value):
        if key in [
            "_wrapper",
            "learning_rate",
            "manager",
            "step",
        ]:
            super().__setattr__(key, value)
        else:
            setattr(self._wrapper.wrapped_optimizer, key, value)

    @property
    def learning_rate(self) -> float:
        """
        :return: convenience function to get the first learning rate for any of
            the param groups in the optimizer
        """
        return get_optim_learning_rate(self._wrapper.wrapped_optimizer)

    @learning_rate.setter
    def learning_rate(self, value: float):
        """
        :param value: the learning rate to set for the optimizer,
            will set all param groups in the optim to this value
        """
        set_optim_learning_rate(self._wrapper.wrapped_optimizer, value)

    @property
    def manager(self) -> ScheduledModifierManager:
        """
        :return: The ScheduledModifierManager for this optimizer
        """
        return self._wrapper.wrapped_manager

    def manager_state_dict(self):
        return self._wrapper.wrapped_manager.state_dict()

    def load_manager_state_dict(self, state_dict):
        self._wrapper.wrapped_manager.load_state_dict(state_dict)

    def step(self, closure=None):
        """
        Called to perform a step on the optimizer activation normal.
        Updates the current epoch based on the step count.
        Calls into modifiers before the step happens.
        Calls into modifiers after the step happens.

        :param closure: optional closure passed into the contained optimizer
            for the step
        """
        self._wrapper.step(closure)

    def loss_update(self, loss: Tensor) -> Tensor:
        """
        Optional call to update modifiers based on the calculated loss.
        Not needed unless one or more of the modifier is using the loss
        to make a modification or is modifying the loss itself.

        :param loss: the calculated loss after running a forward pass and loss_fn
        :return: the modified loss tensor
        """
        return self._wrapper.loss_update(loss)

    def adjust_current_step(self, epoch: int, step: int):
        """
        Adjust the current step for the manager's schedule to the given epoch and step.

        :param epoch: the epoch to set the current global step to match
        :param step: the step (batch) within the epoch to set the
            current global step to match
        """
        warnings.warn(
            "ScheduledOptimizer is deprecated and will be deleted in the future. "
            "adjust_current_step is no longer supported. "
            "Please replace with manager.initialize and manager.modify",
            UserWarning,
        )
