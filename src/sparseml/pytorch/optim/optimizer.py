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

from typing import List, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.utils import (
    PyTorchLogger,
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
    :param loggers: loggers to log important info to within the modifiers;
        ex tensorboard or to the console

    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: Module,
        manager: ScheduledModifierManager,
        steps_per_epoch: int,
        loggers: Union[List[PyTorchLogger], None] = None,
    ):
        # do not call into super since this instance is not passing all calls to
        # the nested optimizer

        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be >= 0")

        self._optimizer = optimizer
        self._module = module
        self._manager = manager
        self._steps_per_epoch = steps_per_epoch
        self._steps = 0

        self._epoch = 0.0
        self._manager.initialize(self._module, self._optimizer)
        self._manager.initialize_loggers(loggers)

    def __del__(self):
        del self._manager

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def __setstate__(self, state):
        self._optimizer.__setstate__(state)

    def __repr__(self):
        self._optimizer.__repr__()

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def __setattr__(self, key, value):
        if key in [
            "_optimizer",
            "_module",
            "_manager",
            "_steps_per_epoch",
            "_steps",
            "_epoch",
            "learning_rate",
            "param_groups",
            "step",
        ]:
            super().__setattr__(key, value)
        else:
            setattr(self._optimizer, key, value)

    @property
    def learning_rate(self) -> float:
        """
        :return: convenience function to get the first learning rate for any of
            the param groups in the optimizer
        """
        return get_optim_learning_rate(self._optimizer)

    @learning_rate.setter
    def learning_rate(self, value: float):
        """
        :param value: the learning rate to set for the optimizer,
            will set all param groups in the optim to this value
        """
        set_optim_learning_rate(self._optimizer, value)

    @property
    def manager(self) -> ScheduledModifierManager:
        """
        :return: The ScheduledModifierManager for this optimizer
        """
        return self._manager

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value

    def state_dict(self):
        return (self._optimizer.state_dict(),)

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def manager_state_dict(self):
        return self._manager.state_dict()

    def load_manager_state_dict(self, state_dict):
        self._manager.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        self._optimizer.add_param_group(param_group)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self, closure=None):
        """
        Called to perform a step on the optimizer activation normal.
        Updates the current epoch based on the step count.
        Calls into modifiers before the step happens.
        Calls into modifiers after the step happens.

        :param closure: optional closure passed into the contained optimizer
            for the step
        """
        self._set_epoch()

        self._manager.update(
            self._module, self._optimizer, self._epoch, self._steps_per_epoch
        )
        self._manager.optimizer_pre_step(
            self._module, self._optimizer, self._epoch, self._steps_per_epoch
        )
        self._optimizer.step(closure)
        self._manager.optimizer_post_step(
            self._module, self._optimizer, self._epoch, self._steps_per_epoch
        )
        self._steps += 1

    def loss_update(self, loss: Tensor) -> Tensor:
        """
        Optional call to update modifiers based on the calculated loss.
        Not needed unless one or more of the modifier is using the loss
        to make a modification or is modifying the loss itself.

        :param loss: the calculated loss after running a forward pass and loss_fn
        :return: the modified loss tensor
        """
        loss = self._manager.loss_update(
            loss, self._module, self._optimizer, self._epoch, self._steps_per_epoch
        )

        return loss

    def adjust_current_step(self, epoch: int, step: int):
        """
        Adjust the current step for the manager's schedule to the given epoch and step.

        :param epoch: the epoch to set the current global step to match
        :param step: the step (batch) within the epoch to set the
            current global step to match
        """
        self._steps = epoch * self._steps_per_epoch + step
        self._set_epoch()
        self._manager.update(
            self._module,
            self._optimizer,
            self._epoch,
            self._steps_per_epoch,
            log_updates=False,
        )

    def _set_epoch(self):
        epoch_num = self._steps // self._steps_per_epoch
        epoch_steps = self._steps % self._steps_per_epoch
        self._epoch = float(epoch_num) + float(epoch_steps) / float(
            self._steps_per_epoch
        )
