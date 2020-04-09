"""
Modifiers for changing the learning rate while training according to
certain update formulas or patterns.
"""

from typing import Dict, Union, List
import sys
import math
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

from neuralmagicML.utils import ALL_TOKEN, convert_to_bool
from neuralmagicML.pytorch.utils.logger import PyTorchLogger
from neuralmagicML.pytorch.recal.modifier import (
    ModifierProp,
    PyTorchModifierYAML,
    ScheduledUpdateModifier,
    ScheduledModifier,
)


__all__ = ["SetLearningRateModifier", "LearningRateModifier"]


CONSTRUCTORS = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
}


def _set_lr(lr: float, optim: Optimizer):
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def _get_lr(optim: Optimizer) -> float:
    for param_group in optim.param_groups:
        return param_group["lr"]

    return -1.0


def _log_lr(
    cur_lr: float, loggers: List[PyTorchLogger], epoch: float, steps_per_epoch: int
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    for logger in loggers:
        logger.log_scalar("Modifier LR", cur_lr, step)


@PyTorchModifierYAML()
class SetLearningRateModifier(ScheduledModifier):
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
        False to only log on an LR change, default True
    """

    def __init__(
        self,
        learning_rate: Union[float, None],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = True,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=None,
        )
        self._learning_rate = learning_rate
        self._lr_set = False
        self._applied = -1.0
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None

    @ModifierProp()
    def learning_rate(self) -> float:
        """
        :return: The learning rate to use once this modifier starts
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: str):
        """
        :param value: The learning rate to use once this modifier starts
        """
        self._learning_rate = value

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
        current_lr = _get_lr(optimizer)

        if self._constant_logging or current_lr != self._last_logged_lr:
            self._last_logged_lr = current_lr
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
            _set_lr(self.learning_rate, optimizer)
            self._applied = self._learning_rate
            self._lr_set = True


@PyTorchModifierYAML()
class LearningRateModifier(ScheduledUpdateModifier):
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
        [StepLR, MultiStepLR, ExponentialLR]
    :param lr_kwargs: The dictionary of keyword arguments to pass to the constructor
        for the lr_class
    :param init_lr: The initial learning rate to use once this modifier starts
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param constant_logging: True to constantly log on every step,
        False to only log on an LR change, default True
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param update_frequency: unused and should not be set
    """

    def __init__(
        self,
        lr_class: str,
        lr_kwargs: Dict,
        init_lr: Union[float, None] = None,
        log_types: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1.0,
            end_comparator=-1,
        )
        self._lr_class = lr_class
        self._lr_kwargs = lr_kwargs
        self._init_lr = init_lr
        self._lr_scheduler = None
        self._base_lr_set = False
        self._last_scheduler_epoch = math.floor(start_epoch)
        self._constant_logging = convert_to_bool(constant_logging)
        self._double_step = False
        self.validate()

    @ModifierProp()
    def lr_class(self) -> str:
        """
        :return: The name of the lr scheduler class to use:
            [StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau]
        """
        return self._lr_class

    @lr_class.setter
    def lr_class(self, value: str):
        """
        :param value: The name of the lr scheduler class to use:
            [StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau]
        """
        self._lr_class = value
        self.validate()

    @ModifierProp()
    def lr_kwargs(self) -> Dict:
        """
        :return: The dictionary of keyword arguments to pass to the constructor
            for the lr_class
        """
        return self._lr_kwargs

    @lr_kwargs.setter
    def lr_kwargs(self, value: Dict):
        """
        :param value: The dictionary of keyword arguments to pass to the constructor
            for the lr_class
        """
        self._lr_kwargs = value

    @ModifierProp()
    def init_lr(self) -> Union[float, None]:
        """
        :return: The initial learning rate to use once this modifier starts
        """
        return self._init_lr

    @init_lr.setter
    def init_lr(self, value: Union[float, None]):
        """
        :param value: The initial learning rate to use once this modifier starts
        """
        self._init_lr = value

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
            self._double_step = True
            return

        if not self._check_setup_lr_scheduler(optimizer, steps_per_epoch):
            self._lr_scheduler.step()

        if self._double_step:
            self._lr_scheduler.step()
            self._double_step = False

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
        current_lr = _get_lr(optimizer)

        if self._constant_logging or current_lr != self._last_logged_lr:
            _log_lr(current_lr, self.loggers, epoch, steps_per_epoch)

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if self._lr_class == "ExponentialLR":
            if "gamma" not in self._lr_kwargs:
                raise ValueError("gamma must be in lr_kwargs for ExponentialLR")
        elif self._lr_class == "StepLR":
            if "gamma" not in self._lr_kwargs:
                raise ValueError("gamma must be in lr_kwargs for StepLR")
            if "step_size" not in self._lr_kwargs:
                raise ValueError("step_size must be in lr_kwargs for StepLR")
        elif self._lr_class == "MultiStepLR":
            if "gamma" not in self._lr_kwargs:
                raise ValueError("gamma must be in lr_kwargs for MultiStepLR")
            if "milestones" not in self._lr_kwargs:
                raise ValueError("milestones must be in lr_kwargs for MultiStepLR")
            for mile in self._lr_kwargs["milestones"]:
                if mile <= self._start_epoch:
                    raise ValueError(
                        "milestones {} all must be greater than start_epoch {}".format(
                            self._lr_kwargs["milestones"], self._start_epoch
                        )
                    )
                if mile >= self._end_epoch:
                    raise ValueError(
                        "milestones {} all must be less than end_epoch {}".format(
                            self._lr_kwargs["milestones"], self._end_epoch
                        )
                    )
        else:
            raise ValueError("unknown lr_class given of {}".format(self._lr_class))

    def _check_init_lr(self, optimizer: Optimizer):
        if self._lr_scheduler is not None:
            return

        if self._init_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self._init_lr

    def _check_setup_lr_scheduler(self, optimizer: Optimizer, steps_per_epoch: int):
        if self._lr_scheduler is not None:
            return False

        if self._lr_class == "ExponentialLR":
            self._lr_kwargs["step_size"] = 1.0
            self._lr_class = "StepLR"

        if self._lr_class == "StepLR":
            self._lr_kwargs["step_size"] = round(
                self._lr_kwargs["step_size"] * steps_per_epoch
            )
        elif self._lr_class == "MultiStepLR":
            self._lr_kwargs["milestones"] = [
                round((mile - self._start_epoch) * steps_per_epoch)
                for mile in self._lr_kwargs["milestones"]
            ]
        else:
            raise ValueError("unrecognized lr_class given of {}".format(self._lr_class))

        self._lr_scheduler = CONSTRUCTORS[self._lr_class](
            optimizer=optimizer, **self._lr_kwargs
        )

        return True
