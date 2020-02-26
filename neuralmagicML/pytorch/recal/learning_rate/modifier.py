"""
Contains code for learning rate modifiers: schedules that change the learning rate while training according to
certain update formulas or patterns
"""

from typing import Dict, Union, List
import sys
import math
import yaml
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CyclicLR

from neuralmagicML.pytorch.utils import ALL_TOKEN, convert_to_bool
from neuralmagicML.pytorch.recal.logger import ModifierLogger
from neuralmagicML.pytorch.recal.modifier import (
    ScheduledUpdateModifier,
    ScheduledModifier,
)


__all__ = ["SetLearningRateModifier", "LearningRateModifier", "CyclicLRModifier"]


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
    cur_lr: float, loggers: List[ModifierLogger], epoch: float, steps_per_epoch: int
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    for logger in loggers:
        logger.log_scalar("Modifier LR", cur_lr, step)


class SetLearningRateModifier(ScheduledModifier):
    """
    Modifier to set the learning rate to a specific value at a certain point in the training process
    Once that point is reached, will update the optimizer's params with the learning rate

    Sample yaml:
        !SetLearningRateModifier
            start_epoch: 0.0
            learning_rate: 0.001
            allowed_loggers: __ALL__
            constant_logging: True
    """

    YAML_KEY = u"!SetLearningRateModifier"

    @staticmethod
    def yaml_constructor(loader, node):
        """
        Create an instance of the modifier from a yaml file
        Follows the yaml package in python for implementation and integration
        """
        instance = SetLearningRateModifier.__new__(SetLearningRateModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(
        self,
        learning_rate: Union[float, None],
        start_epoch: float = -1.0,
        allowed_loggers: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = True,
    ):
        """
        :param learning_rate: The learning rate to use once this modifier starts
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param allowed_loggers: The loggers to allow the learning rate to be logged to, default is __ALL__
        :param constant_logging: True to constantly log on every step, False to only log on an LR change, default True
        """
        super().__init__(start_epoch, end_epoch=-1.0, allowed_loggers=allowed_loggers)
        self._learning_rate = learning_rate
        self._lr_set = False
        self._applied = -1.0
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None

    def __repr__(self):
        return "{}(learning_rate={}, start_epoch={})".format(
            self.__class__.__name__, self.learning_rate, self.start_epoch
        )

    @property
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
        if self._initialized:
            raise RuntimeError(
                "Cannot change learning_rate after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._learning_rate = value

    @property
    def applied_learning_rate(self) -> float:
        """
        :return: the last applied learning rate to the optimizer, -1.0 if hasn't been applied
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
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
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
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
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


yaml.add_constructor(
    SetLearningRateModifier.YAML_KEY, SetLearningRateModifier.yaml_constructor
)
yaml.add_constructor(
    SetLearningRateModifier.YAML_KEY,
    SetLearningRateModifier.yaml_constructor,
    yaml.SafeLoader,
)


class LearningRateModifier(ScheduledUpdateModifier):
    """
    Modifier to set the learning rate to specific values at certain points in the training process between set epochs
    Any time an update point is reached, the LR is updated for the parameters in the optimizer
    Builds on top of the builtin LR schedulers in pytorch

    Sample yaml:
        !LearningRateModifier
            start_epoch: 0.0
            end_epoch: 10.0
            update_frequency: 1.0
            lr_class: ExponentialLR
            lr_kwargs:
                gamma: 0.95
            init_lr: 0.01
            allowed_loggers: __ALL__
            constant_logging: True
    """

    YAML_KEY = u"!LearningRateModifier"

    @staticmethod
    def yaml_constructor(loader, node):
        """
        Create an instance of the modifier from a yaml file
        Follows the yaml package in python for implementation and integration
        """
        instance = LearningRateModifier.__new__(LearningRateModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(
        self,
        lr_class: str,
        lr_kwargs: Dict,
        init_lr: Union[float, None] = None,
        allowed_loggers: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = 1.0,
    ):
        """
        :param lr_class: The name of the lr scheduler class to use: [StepLR, MultiStepLR, ExponentialLR]
        :param lr_kwargs: The dictionary of keyword arguments to pass to the constructor for the lr_class
        :param init_lr: The initial learning rate to use once this modifier starts
        :param allowed_loggers: The loggers to allow the learning rate to be logged to, default is __ALL__
        :param constant_logging: True to constantly log on every step, False to only log on an LR change, default True
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        """
        super().__init__(start_epoch, end_epoch, update_frequency, allowed_loggers)
        self._lr_class = lr_class
        self._lr_kwargs = lr_kwargs
        self._init_lr = init_lr
        self._lr_scheduler = None
        self._base_lr_set = False
        self._last_scheduler_epoch = math.floor(start_epoch)
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None

        if "milestones" in self._lr_kwargs:
            self._lr_kwargs["milestones"] = [
                mile - self._start_epoch for mile in self._lr_kwargs["milestones"]
            ]

        assert self._lr_class in CONSTRUCTORS

    def __repr__(self):
        return "{}(lr_class={}, lr_kwargs={}, init_lr={}, start_epoch={}, end_epoch={}, update_frequency={})".format(
            self.__class__.__name__,
            self.lr_class,
            self.lr_kwargs,
            self.init_lr,
            self.start_epoch,
            self.end_epoch,
            self.update_frequency,
        )

    @property
    def lr_class(self) -> str:
        """
        :return: The name of the lr scheduler class to use: [StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau]
        """
        return self._lr_class

    @lr_class.setter
    def lr_class(self, value: str):
        """
        :param value: The name of the lr scheduler class to use: [StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau]
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change lr_class after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._lr_class = value

    @property
    def lr_kwargs(self) -> Dict:
        """
        :return: The dictionary of keyword arguments to pass to the constructor for the lr_class
        """
        return self._lr_kwargs

    @lr_kwargs.setter
    def lr_kwargs(self, value: Dict):
        """
        :param value: The dictionary of keyword arguments to pass to the constructor for the lr_class
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change lr_kwargs after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._lr_kwargs = value

    @property
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
        if self._initialized:
            raise RuntimeError(
                "Cannot change init_lr after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._init_lr = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Create the lr_scheduler using the optimizer and the provided lr_kwargs

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(LearningRateModifier, self).initialize(module, optimizer)
        self._lr_scheduler = CONSTRUCTORS[self._lr_class](
            optimizer=optimizer, **self._lr_kwargs
        )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Calls into the lr scheduler to step given the epoch
        Additionally will first set the lr to the init_lr if not set yet

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_setup_base_lrs(optimizer, epoch)

        if epoch < sys.float_info.epsilon or (
            epoch < self.start_epoch
            and abs(epoch - self.start_epoch) > sys.float_info.epsilon
        ):
            return

        if math.floor(epoch) != self._last_scheduler_epoch:
            self._lr_scheduler.step()
            self._last_scheduler_epoch = math.floor(epoch)

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
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)
        current_lr = _get_lr(optimizer)

        if self._constant_logging or current_lr != self._last_logged_lr:
            self._last_logged_lr = current_lr
            _log_lr(current_lr, self.loggers, epoch, steps_per_epoch)

    def _check_setup_base_lrs(self, optimizer: Optimizer, epoch: float):
        if (
            self.start_epoch < 0.0
            or (self.start_epoch - epoch) < sys.float_info.epsilon
        ) and not self._base_lr_set:
            if self._init_lr is not None:
                self._lr_scheduler.base_lrs = list(
                    map(lambda group: self._init_lr, optimizer.param_groups)
                )

                _set_lr(self._init_lr, optimizer)
            else:
                self._lr_scheduler.base_lrs = list(
                    map(lambda group: group["lr"], optimizer.param_groups)
                )

            self._base_lr_set = True


yaml.add_constructor(
    LearningRateModifier.YAML_KEY, LearningRateModifier.yaml_constructor
)
yaml.add_constructor(
    LearningRateModifier.YAML_KEY,
    LearningRateModifier.yaml_constructor,
    yaml.SafeLoader,
)


class CyclicLRModifier(ScheduledUpdateModifier):
    """
    Modifier to set the learning rate based on a cyclic LR schedule between set epochs
    Any time an update point is reached, the LR is updated for the parameters in the optimizer
    Builds on top of the builtin cyclic LR scheduler in pytorch

    Sample yaml:
        !CyclicLRModifier
            start_epoch: 0.0
            end_epoch: 10.0
            base_lr: 0.0001
            max_lr: 0.01
            allowed_loggers: __ALL__
            constant_logging: True
    """

    YAML_KEY = u"!CyclicLRModifier"

    @staticmethod
    def yaml_constructor(loader, node):
        instance = CyclicLRModifier.__new__(CyclicLRModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(
        self,
        lr_kwargs: Dict,
        allowed_loggers: Union[str, List[str]] = ALL_TOKEN,
        constant_logging: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
    ):
        """
        :param lr_kwargs: The dictionary of keyword arguments to pass to the constructor for the lr_class
        :param allowed_loggers: The loggers to allow the learning rate to be logged to, default is __ALL__
        :param constant_logging: True to constantly log on every step, False to only log on an LR change, default True
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        """
        super().__init__(
            start_epoch,
            end_epoch,
            update_frequency=-1.0,
            allowed_loggers=allowed_loggers,
        )
        self._lr_kwargs = lr_kwargs
        self._lr_scheduler = None
        self._lr_set = False
        self._constant_logging = convert_to_bool(constant_logging)
        self._last_logged_lr = None

        assert "base_lr" in lr_kwargs
        assert "max_lr" in lr_kwargs

    def __repr__(self):
        return "{}(lr_kwargs={}, start_epoch={}, end_epoch={})".format(
            self.__class__.__name__, self._lr_kwargs, self.start_epoch, self.end_epoch
        )

    @property
    def lr_kwargs(self) -> Dict:
        """
        :return: the key word args that are passed to the cyclic scheduler class,
                 includes most of the params given in the constructor
        """
        return self._lr_kwargs

    @lr_kwargs.setter
    def lr_kwargs(self, value: Dict):
        """
        :param value: the key word args that are passed to the cyclic scheduler class,
                      includes most of the params given in the constructor
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change lr_kwargs after {} has been initialized".format(
                    self.__class__.__name__
                )
            )

        self._lr_kwargs = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Create the lr_scheduler using the optimizer and the provided lr_kwargs

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(CyclicLRModifier, self).initialize(module, optimizer)
        init_lrs = [g["lr"] for g in optimizer.param_groups]
        self._lr_scheduler = CyclicLR(optimizer=optimizer, **self._lr_kwargs)
        for group, init_lr in zip(optimizer.param_groups, init_lrs):
            group["lr"] = init_lr

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Calls into the lr scheduler to step for each batch

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_set_lr(optimizer, epoch)

        if epoch < sys.float_info.epsilon or (
            epoch < self.start_epoch
            and abs(epoch - self.start_epoch) > sys.float_info.epsilon
        ):
            return

        batch_count = round(epoch * steps_per_epoch)
        self._lr_scheduler.step(batch_count)

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
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)
        current_lr = _get_lr(optimizer)

        if self._constant_logging or current_lr != self._last_logged_lr:
            self._last_logged_lr = current_lr
            _log_lr(current_lr, self.loggers, epoch, steps_per_epoch)

    def _check_set_lr(self, optimizer: Optimizer, epoch: float):
        if (
            self.start_epoch < 0.0
            or (self.start_epoch - epoch) < sys.float_info.epsilon
        ) and not self._lr_set:
            _set_lr(self.lr_kwargs["base_lr"], optimizer)
            self._lr_set = True


yaml.add_constructor(CyclicLRModifier.YAML_KEY, CyclicLRModifier.yaml_constructor)
yaml.add_constructor(
    CyclicLRModifier.YAML_KEY, CyclicLRModifier.yaml_constructor, yaml.SafeLoader
)
