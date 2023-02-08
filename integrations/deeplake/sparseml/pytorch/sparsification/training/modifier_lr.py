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
from typing import Dict, List, Optional, Union

from torch.nn import Module
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    StepLR,
)
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier
from sparseml.pytorch.sparsification.modifier import (
    ModifierProp,
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import (
    get_optim_groups_learning_rates,
    set_optim_learning_rate,
)
from sparseml.sparsification import LearningRateModifier as BaseLearningRateModifier
from sparseml.sparsification import (
    SetLearningRateModifier as BaseSetLearningRateModifier,
)
from sparseml.sparsification import SparsificationTypes
from sparseml.utils import convert_to_bool


__all__ = [
    "SetLearningRateModifier",
    "LearningRateFunctionModifier",
    "LearningRateModifier",
]


CONSTRUCTORS = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}


@PyTorchModifierYAML()
class SetLearningRateModifier(BaseSetLearningRateModifier, ScheduledModifier):
    """
    Modifier to set the learning rate to a specific value at a certain point in the
    training process.
    Once that point is reached,
    will update the optimizer's params with the learning rate.

    | Sample yaml:
    |   !SetLearningRateModifier
    |       start_epoch: 0.0
    |       learning_rate: 0.001
    |       constant_logging: True

    :param learning_rate: The learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: unused and should not be set
    :param constant_logging: True to constantly log on every step,
        False to only log on an LR change and min once per epoch, default False
    """

    def __init__(
        self,
        learning_rate: Union[float, None],
        param_groups: Optional[List[int]] = None,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        constant_logging: bool = False,
    ):
        super(SetLearningRateModifier, self).__init__(
            learning_rate=learning_rate,
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=None,
        )
        self._param_groups = param_groups
        self._lr_set = False
        self._applied = -1.0
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None

    @ModifierProp()
    def param_groups(self) -> Optional[List[int]]:
        """
        :return: The param group indices to set the lr for within the optimizer,
            if not set will set the lr for all param groups
        """
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value: Optional[List[int]]):
        """
        :param value: The param group indices to set the lr for within the optimizer,
            if not set will set the lr for all param groups
        """
        self._param_groups = value

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

    @ScheduledModifier.log_call
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
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
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
        group_lrs = [
            (f"ParamGroup{index}", lr)
            for (index, lr) in enumerate(get_optim_groups_learning_rates(optimizer))
            if not self.param_groups or index in self.param_groups
        ]

        if not group_lrs:
            raise ValueError(
                "Could not find param groups in the optimizer "
                f"for given param_groups {self.param_groups}"
            )

        current_lr = group_lrs[-1][1]

        if self._constant_logging or self._last_logged_lr != current_lr:
            self._last_logged_lr = current_lr
            self.log_named_scalars(
                name_value_pairs=group_lrs,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

    def _check_set_lr(self, optimizer: Optimizer, epoch: float):
        if (
            (
                self.start_epoch < 0.0
                or (self.start_epoch - epoch) < sys.float_info.epsilon
            )
            and not self._lr_set
            and self._learning_rate is not None
        ):
            for (index, group) in enumerate(optimizer.param_groups):
                if not self.param_groups or index in self.param_groups:
                    group["lr"] = self.learning_rate
            self._applied = self.learning_rate
            self._lr_set = True


@PyTorchModifierYAML()
class LearningRateFunctionModifier(ScheduledUpdateModifier):
    """
    Modifier to set the learning rate based on supported math functions scaling between
    an init_lr and a final_lr.
    Any time an update point is reached, the LR is updated for the parameters groups
    in the optimizer.
    Specific parameter groups can be targeted for the optimizer as well.

    | Sample yaml:
    |   !LearningRateFunctionModifier
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       lr_func: linear
    |       init_lr: 0.1
    |       final_lr: 0.001

    :param lr_func: The name of the lr function to use: [linear, cosine]
    :param init_lr: The initial learning rate to use once this modifier starts
    :param init_lr: The final learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: The epoch to end the modifier at,
        (set to -1.0 so it doesn't end)
    :param cycle_epochs: The number of epochs between two consecutive LR rewinding;
        used for cyclic_linear schedule only.
    :param_groups: The param group indices to set the lr for within the optimizer,
        if not set will set the lr for all param groups
    :param update_frequency: unused and should not be set
    :param constant_logging: True to constantly log on every step,
        False to only log on an LR change and min once per epoch, default False
    """

    def __init__(
        self,
        lr_func: str,
        init_lr: float,
        final_lr: float,
        start_epoch: float,
        end_epoch: float,
        cycle_epochs: float = 1.0,
        param_groups: Optional[List[int]] = None,
        update_frequency: float = -1.0,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1.0,
            end_comparator=1,
        )
        self._lr_func = lr_func
        self._init_lr = init_lr
        self._final_lr = final_lr
        self._cycle_epochs = cycle_epochs
        self._param_groups = param_groups
        self._learning_rate = None
        self._last_applied_lr = None
        self._last_logged_lr = None
        self.validate()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.learning_rate]

    @ModifierProp()
    def lr_func(self) -> str:
        """
        :return: The name of the lr function to use: [linear, cosine]
        """
        return self._lr_func

    @lr_func.setter
    def lr_func(self, value: str):
        """
        :param value: The name of the lr function to use: [linear, cosine]
        """
        self._lr_func = value
        self.validate()

    @ModifierProp()
    def init_lr(self) -> float:
        """
        :return: The initial learning rate to use once this modifier starts
        """
        return self._init_lr

    @init_lr.setter
    def init_lr(self, value: float):
        """
        :param value: The initial learning rate to use once this modifier starts
        """
        self._init_lr = value
        self.validate()

    @ModifierProp()
    def final_lr(self) -> float:
        """
        :return: The final learning rate to use once this modifier starts
        """
        return self._final_lr

    @final_lr.setter
    def final_lr(self, value: float):
        """
        :param value: The final learning rate to use once this modifier starts
        """
        self._final_lr = value
        self.validate()

    @ModifierProp()
    def cycle_epochs(self) -> float:
        return self._cycle_epochs

    @cycle_epochs.setter
    def cycle_epochs(self, value: float):
        self._cycle_epochs = value
        self.validate()

    @ModifierProp()
    def param_groups(self) -> Optional[List[int]]:
        """
        :return: The param group indices to set the lr for within the optimizer,
            if not set will set the lr for all param groups
        """
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value: Optional[List[int]]):
        """
        :param value: The param group indices to set the lr for within the optimizer,
            if not set will set the lr for all param groups
        """
        self._param_groups = value
        self.validate()

    @ScheduledModifier.log_call
    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Updates the LR based on the given epoch for the optimizer

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        lambad_func = getattr(LearningRateFunctionModifier, f"_{self._lr_func}")
        self._learning_rate = lambad_func(self, epoch, steps_per_epoch)
        set_optim_learning_rate(optimizer, self._learning_rate, self.param_groups)

    def log_update(
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ):
        """
        Check whether to log an update for the learning rate of the modifier.
        Checks for a change in the LR or epoch before logging

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)
        group_lrs = [
            (f"ParamGroup{index}", lr)
            for (index, lr) in enumerate(get_optim_groups_learning_rates(optimizer))
            if not self.param_groups or index in self.param_groups
        ]

        if not group_lrs:
            raise ValueError(
                "Could not find param groups in the optimizer "
                f"for given param_groups {self.param_groups}"
            )

        current_lr = group_lrs[-1][1]

        if current_lr != self._last_logged_lr:
            self.log_named_scalars(
                name_value_pairs=group_lrs,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )
            self._last_logged_lr = current_lr

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """
        lr_funcs = ["linear", "cosine", "cyclic_linear"]
        if self.lr_func not in lr_funcs:
            raise ValueError(f"lr_func must be one of {lr_funcs}")

        if lr_funcs == "cyclic_linear" and self.cycle_epochs <= 0.0:
            raise ValueError(
                "cycle_epochs in the cyclic_linear schedule must be positive"
            )

        if isinstance(self.init_lr, str):
            self.init_lr = float(self.init_lr)

        if (
            (not self.init_lr and self.init_lr != 0)
            or self.init_lr < 0.0
            or self.init_lr > 1.0
        ):
            raise ValueError(
                f"init_lr must be within range [0.0, 1.0], given {self.init_lr}"
            )

        if isinstance(self.final_lr, str):
            self.final_lr = float(self.final_lr)

        if (
            (not self.final_lr and self.final_lr != 0)
            or self.final_lr < 0.0
            or self.final_lr > 1.0
        ):
            raise ValueError(
                f"final_lr must be within range [0.0, 1.0], given {self.final_lr}"
            )

        if self.update_frequency != -1.0:
            raise ValueError("update_frequency must be kept at -1.0")

    def _linear(self, epoch: float, steps_per_epoch: int) -> float:
        # y = y1 + ((x – x1) / (x2 – x1)) * (y2 – y1)
        start = self.start_epoch if self.start_epoch > 0 else 0.0
        end = self.end_epoch

        return self.init_lr + ((epoch - start) / (end - start)) * (
            self.final_lr - self.init_lr
        )

    def _cosine(self, epoch: float, steps_per_epoch: int) -> float:
        start = self.start_epoch if self.start_epoch > 0 else 0.0
        end = self.end_epoch

        # scale x to [0-1] for use with cosine
        x_norm = (epoch - start) / (end - start)

        # conditional to support cosine down to a value and up to a value
        if self.final_lr < self.init_lr:
            y_range = self.init_lr - self.final_lr
            y_shift = self.final_lr
            x_shift = 0
        else:
            y_range = self.final_lr - self.init_lr
            y_shift = self.init_lr
            x_shift = math.pi

        return (
            math.cos(x_norm * math.pi + x_shift) * y_range / 2 + y_range / 2 + y_shift
        )

    def _cyclic_linear(self, epoch: float, steps_per_epoch: int):
        end_step = self.end_epoch * steps_per_epoch
        start_step = self.start_epoch * steps_per_epoch
        cycle_steps = self.cycle_epochs * steps_per_epoch
        current_step = (epoch - self.start_epoch) * steps_per_epoch
        if current_step > int((end_step - start_step) / cycle_steps) * cycle_steps:
            cycle_steps = (end_step - start_step) % cycle_steps
        adjusted_step = current_step % cycle_steps
        lr = self.init_lr - (adjusted_step / (cycle_steps - 1)) * (
            self.init_lr - self.final_lr
        )
        return lr


@PyTorchModifierYAML()
class LearningRateModifier(BaseLearningRateModifier, ScheduledUpdateModifier):
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
        constant_logging: bool = False,
    ):
        super(LearningRateModifier, self).__init__(
            lr_class=lr_class,
            lr_kwargs=lr_kwargs,
            init_lr=init_lr,
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

    @ScheduledModifier.log_call
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
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
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
        group_lrs = [
            (f"ParamGroup{index}", lr)
            for (index, lr) in enumerate(get_optim_groups_learning_rates(optimizer))
        ]

        if not group_lrs:
            raise ValueError("Could not find any param groups in the optimizer")

        current_lr = group_lrs[-1][1]

        if self._constant_logging or current_lr != self._last_logged_lr:
            self._last_logged_lr = current_lr
            self.log_named_scalars(
                name_value_pairs=group_lrs,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

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
