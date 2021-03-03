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
Modifiers for changing the learning rate while training according to
certain update formulas or patterns.
"""

import math
import sys
from typing import Dict, List, Union

from torch.nn import Module
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    StepLR,
)
from torch.optim.optimizer import Optimizer

from sparseml.optim import LearningRate, SetLearningRate
from sparseml.pytorch.optim.modifier import (
    ModifierProp,
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import (
    PyTorchLogger,
    get_optim_learning_rate,
    set_optim_learning_rate,
)
from sparseml.utils import ALL_TOKEN, convert_to_bool


__all__ = ["SetLearningRateModifier", "LearningRateModifier"]


CONSTRUCTORS = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}


def _log_lr(
    cur_lr: float, loggers: List[PyTorchLogger], epoch: float, steps_per_epoch: int
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    for logger in loggers:
        logger.log_scalar("Modifier LR", cur_lr, step)


@PyTorchModifierYAML()
class SetLearningRateModifier(ScheduledModifier, SetLearningRate):
    """
    Modifier to set the learning rate to a specific value at a certain point in the
    training process.
    Once that point is reached,
    will update the optimizer's params with the learning rate.

    | Sample yaml:
    |   !SetLearningRateModifier
    |       start_epoch: 0.0
    |       learning_rate: 0.001
    |       log_types: __ALL__
    |       constant_logging: True

    :param learning_rate: The learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: unused and should not be set
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param constant_logging: True to constantly log on every step,
        False to only log on an LR change and min once per epoch, default False
    """

    def __init__(
        self,
        learning_rate: Union[float, None],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=None,
        )
        self._lr_set = False
        self._applied = -1.0
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None
        self._last_logged_epoch = None

    @ModifierProp()
    def constant_logging(self) -> bool:
        """
        :return: True to constantly log on every step,
            False to only log on an LR change, default True
        """
        return self._constant_logging

    @constant_logging.setter
    def constant_logging(self, value: bool):
        """
        :param value: True to constantly log on every step,
            False to only log on an LR change, default True
        """
        self._constant_logging = value

    @ModifierProp(serializable=False)
    def applied_learning_rate(self) -> float:
        """
        :return: the last applied learning rate to the optimizer,
            -1.0 if hasn't been applied
        """
        return self._applied

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to update the learning rate for the optimizer or not

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_set_lr(optimizer, epoch)

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to log an update for the learning rate of the modifier
        If constant logging is enabled, then will always log
        Otherwise checks for a change in the LR before logging

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)
        current_lr = get_optim_learning_rate(optimizer)

        if (
            self._constant_logging
            or current_lr != self._last_logged_lr
            or math.floor(epoch) != self._last_logged_epoch
        ):
            self._last_logged_lr = current_lr
            self._last_logged_epoch = math.floor(epoch)
            _log_lr(current_lr, self.loggers, epoch, steps_per_epoch)

    def _check_set_lr(self, optimizer: Optimizer, epoch: float):
        if (
            (
                self.start_epoch < 0.0
                or (self.start_epoch - epoch) < sys.float_info.epsilon
            )
            and not self._lr_set
            and self._learning_rate is not None
        ):
            set_optim_learning_rate(optimizer, self.learning_rate)
            self._applied = self._learning_rate
            self._lr_set = True


@PyTorchModifierYAML()
class LearningRateModifier(ScheduledUpdateModifier, LearningRate):
    """
    Modifier to set the learning rate to specific values at certain points in the
    training process between set epochs.
    Any time an update point is reached, the LR is updated for the parameters
    in the optimizer.
    Builds on top of the builtin LR schedulers in PyTorch.

    | Sample yaml:
    |   !LearningRateModifier
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       lr_class: ExponentialLR
    |       lr_kwargs:
    |           gamma: 0.95
    |       init_lr: 0.01
    |       log_types: __ALL__
    |       constant_logging: True

    :param lr_class: The name of the lr scheduler class to use:
        [StepLR, MultiStepLR, ExponentialLR, CosineAnnealingWarmRestarts]
    :param lr_kwargs: The dictionary of keyword arguments to pass to the constructor
        for the lr_class
    :param init_lr: The initial learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: The epoch to end the modifier at,
        (set to -1.0 so it doesn't end)
    :param update_frequency: unused and should not be set
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param constant_logging: True to constantly log on every step,
        False to only log on an LR change and min once per epoch, default False
    """

    def __init__(
        self,
        lr_class: str,
        lr_kwargs: Dict,
        init_lr: float,
        start_epoch: float,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = False,
    ):
        super().__init__(
            lr_class=lr_class,
            lr_kwargs=lr_kwargs,
            init_lr=init_lr,
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1.0,
            end_comparator=-1,
        )
        self._lr_scheduler = None
        self._base_lr_set = False
        self._last_scheduler_epoch = math.floor(start_epoch)
        self._constant_logging = convert_to_bool(constant_logging)
        self._double_step = False
        self._last_logged_lr = None
        self._last_logged_epoch = None
        self._scheduler_steps = 0
        self.validate()

    @ModifierProp()
    def constant_logging(self) -> bool:
        """
        :return: True to constantly log on every step,
            False to only log on an LR change, default True
        """
        return self._constant_logging

    @constant_logging.setter
    def constant_logging(self, value: bool):
        """
        :param value: True to constantly log on every step,
            False to only log on an LR change, default True
        """
        self._constant_logging = value

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Calls into the lr scheduler to step given the epoch
        Additionally will first set the lr to the init_lr if not set yet

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_init_lr(optimizer)

        if epoch <= sys.float_info.epsilon:
            # make sure we don't apply an lr step before the optimizer step
            # mark the step to be applied on the next update
            self._scheduler_steps -= 1
            return

        if (
            abs(self.end_epoch - epoch) <= sys.float_info.epsilon
            and self.end_epoch >= 0.0
        ):
            # no cleanup step for LR, so exit before adding another LR step
            return

        self._check_setup_lr_scheduler(optimizer, epoch, steps_per_epoch)

        if self.lr_class != "CosineAnnealingWarmRestarts":
            global_step = (
                round(epoch * steps_per_epoch)
                if self.end_epoch < 0.0 or epoch <= self.end_epoch
                else round(self.end_epoch * steps_per_epoch)
            )
            step_diff = global_step - self._scheduler_steps

            if step_diff > 0:
                for _ in range(step_diff):
                    self._lr_scheduler.step()

                self._scheduler_steps = global_step
        else:
            self._lr_scheduler.step(
                epoch - self.start_epoch if self.start_epoch >= 0.0 else 0.0
            )

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to log an update for the learning rate of the modifier
        If constant logging is enabled, then will always log
        Otherwise checks for a change in the LR before logging

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)
        current_lr = get_optim_learning_rate(optimizer)

        if (
            self._constant_logging
            or current_lr != self._last_logged_lr
            or math.floor(epoch) != self._last_logged_epoch
        ):
            self._last_logged_lr = current_lr
            self._last_logged_epoch = math.floor(epoch)
            _log_lr(current_lr, self.loggers, epoch, steps_per_epoch)

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if self.update_frequency != -1.0:
            raise ValueError("update_frequency must be kept at -1.0")

    def _check_init_lr(self, optimizer: Optimizer):
        if self._lr_scheduler is not None:
            return

        if self._init_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self._init_lr

    def _check_setup_lr_scheduler(
        self, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        if self._lr_scheduler is not None:
            return False

        lr_class, lr_kwargs = self.corrected_lr_info(
            steps_per_epoch, self.start_epoch, self.end_epoch
        )
        self._lr_scheduler = CONSTRUCTORS[lr_class](optimizer=optimizer, **lr_kwargs)
        if hasattr(optimizer, "_step_count"):
            # hack to keep pytorch lr scheduler from complaining
            optimizer._step_count += 1

        global_step = round(epoch * steps_per_epoch)
        self._scheduler_steps += global_step

        return True
