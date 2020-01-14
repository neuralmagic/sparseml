from typing import Dict, Union, Callable
import sys
import yaml
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR
)

from .modifier import ScheduledUpdateModifier, ScheduledModifier


__all__ = ['LearningRateModifier', 'CyclicLRModifier']


CONSTRUCTORS = {
    'StepLR': StepLR,
    'MultiStepLR': MultiStepLR,
    'ExponentialLR': ExponentialLR,
    'ReduceLROnPlateau': ReduceLROnPlateau
}


class SetLearningRateModifier(ScheduledModifier):
    YAML_KEY = u'!SetLearningRateModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = SetLearningRateModifier.__new__(SetLearningRateModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, learning_rate: Union[float, None], start_epoch: float = -1.0):
        """
        Controls the learning rate of the optimizer based on a scheduled frequency

        Sample yaml:
            !SetLearningRateModifier
                start_epoch: 0.0
                learning_rate: 0.001

        :param learning_rate: The learning rate to use once this modifier starts
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        """
        super().__init__(start_epoch, end_epoch=-1.0)
        self._learning_rate = learning_rate
        self._lr_set = False

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: str):
        if self._initialized:
            raise RuntimeError('Cannot change learning_rate after {} has been initialized'
                               .format(self.__class__.__name__))

        self._learning_rate = value

    def initialize(self, module: Module, optimizer: Optimizer):
        super().initialize(module, optimizer)
        self._check_set_lr(optimizer, 0.0)

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        # not needed
        pass

    def optimizer_pre_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Sets the initial lr

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        self._check_set_lr(optimizer, epoch)

    def optimizer_post_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Sets the initial lr

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        self._check_set_lr(optimizer, epoch)
        self._lr_set = True

    def _check_set_lr(self, optimizer: Optimizer, epoch: float):
        if ((self.start_epoch < 0.0 or (self.start_epoch - epoch) < sys.float_info.epsilon)
                and not self._lr_set and self._learning_rate is not None):
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate


yaml.add_constructor(SetLearningRateModifier.YAML_KEY, SetLearningRateModifier.yaml_constructor)
yaml.add_constructor(SetLearningRateModifier.YAML_KEY, SetLearningRateModifier.yaml_constructor, yaml.SafeLoader)


class LearningRateModifier(ScheduledUpdateModifier):
    YAML_KEY = u'!LearningRateModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = LearningRateModifier.__new__(LearningRateModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, lr_class: str, lr_kwargs: Dict, init_lr: Union[float, None] = None, adjust_update: bool = True,
                 start_epoch: float = -1.0, end_epoch: float = -1.0, update_frequency: float = 1.0):
        """
        Controls the learning rate of the optimizer based on a scheduled frequency

        Sample yaml:
            !LearningRateModifier
                start_epoch: 0.0
                end_epoch: 10.0
                update_frequency: 1.0
                lr_class: ExponentialLR
                lr_kwargs:
                    gamma: 0.95
                init_lr: 0.01

        :param lr_class: The name of the lr scheduler class to use:
                         [StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts]
        :param lr_kwargs: The dictionary of keyword arguments to pass to the constructor for the lr_class
        :param adjust_update: Adjust the update steps down by the start epoch so the first is 0 rather than start epoch
        :param init_lr: The initial learning rate to use once this modifier starts
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        """
        super().__init__(start_epoch, end_epoch, update_frequency)
        self._lr_class = lr_class
        self._lr_kwargs = lr_kwargs
        self._adjust_update = adjust_update
        self._init_lr = init_lr
        self._lr_scheduler = None
        self._init_lr_set = False

        assert self._lr_class in CONSTRUCTORS

    @property
    def lr_class(self) -> str:
        return self._lr_class

    @lr_class.setter
    def lr_class(self, value: str):
        if self._initialized:
            raise RuntimeError('Cannot change lr_class after {} has been initialized'
                               .format(self.__class__.__name__))

        self._lr_class = value

    @property
    def lr_kwargs(self) -> Dict:
        return self._lr_kwargs

    @lr_kwargs.setter
    def lr_kwargs(self, value: Dict):
        if self._initialized:
            raise RuntimeError('Cannot change lr_kwargs after {} has been initialized'
                               .format(self.__class__.__name__))

        self._lr_kwargs = value

    @property
    def init_lr(self) -> Union[float, None]:
        return self._init_lr

    @init_lr.setter
    def init_lr(self, value: Union[float, None]):
        if self._initialized:
            raise RuntimeError('Cannot change init_lr after {} has been initialized'
                               .format(self.__class__.__name__))

        self._init_lr = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Create the lr_scheduler using the optimizer and the provided lr_kwargs

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(LearningRateModifier, self).initialize(module, optimizer)
        self._lr_scheduler = CONSTRUCTORS[self._lr_class](optimizer=optimizer, **self._lr_kwargs)

        if self._init_lr is not None:
            self._lr_scheduler.base_lrs = list(map(lambda group: self._init_lr, optimizer.param_groups))

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Calls into the lr scheduler to step given the epoch
        Additionally will first set the lr to the init_lr if not set yet

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """

        if epoch < sys.float_info.epsilon:
            # will not step on first update step (before optimizer is called), because of implementation detail
            # use the initial lr instead to set the first value
            return

        if self._adjust_update:
            epoch = epoch - self.start_epoch

        self._lr_scheduler.step(epoch)

    def optimizer_pre_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Sets the initial lr if given

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        self._check_set_lr(optimizer, epoch)

    def optimizer_post_step(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Sets the initial lr if given

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        self._check_set_lr(optimizer, epoch)
        self._init_lr_set = True

    def _check_set_lr(self, optimizer: Optimizer, epoch: float):
        if ((self.start_epoch < 0.0 or (self.start_epoch - epoch) < sys.float_info.epsilon)
                and not self._init_lr_set and self._init_lr is not None):
            for param_group in optimizer.param_groups:
                param_group['lr'] = self._init_lr


yaml.add_constructor(LearningRateModifier.YAML_KEY, LearningRateModifier.yaml_constructor)
yaml.add_constructor(LearningRateModifier.YAML_KEY, LearningRateModifier.yaml_constructor, yaml.SafeLoader)


class CyclicLRModifier(ScheduledUpdateModifier):
    YAML_KEY = u'!CyclicLRModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = LearningRateModifier.__new__(LearningRateModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, base_lr: float, max_lr: float, step_size_up: int = 2000, step_size_down: Union[int, None] = None,
                 mode: str = 'triangular', gamma: float = 1.0, scale_fn: Callable = None, scale_mode: str = 'cycle',
                 cycle_momentum: bool = True, base_momentum: float = 0.8, max_momentum: float = 0.9,
                 start_epoch: float = -1.0, end_epoch: float = -1.0):
        """
        Controls the learning rate of the optimizer based on the cyclic LR

        Sample yaml:
            !CyclicLRModifier
                start_epoch: 0.0
                end_epoch: 10.0
                base_lr: 0.0001
                max_lr: 0.01

        :param base_lr: Initial learning rate which is the lower boundary in the cycle for each parameter group.
        :param max_lr: Upper learning rate boundaries in the cycle for each parameter group.
                       Functionally, it defines the cycle amplitude (max_lr - base_lr).
                       The lr at any cycle is the sum of base_lr and some scaling of the amplitude;
                       therefore max_lr may not actually be reached depending on scaling function.
        :param step_size_up: Number of training iterations in the increasing half of a cycle. Default: 2000
        :param step_size_down: Number of training iterations in the decreasing half of a cycle.
                               If step_size_down is None, it is set to step_size_up. Default: None
        :param mode: One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above.
                     If scale_fn is not None, this argument is ignored. Default: 'triangular'
        :param gamma: Constant in 'exp_range' scaling function: gamma**(cycle iterations) Default: 1.0
        :param scale_fn: Custom scaling policy defined by a single argument lambda function,
                         where 0 <= scale_fn(x) <= 1 for all x >= 0. If specified, then 'mode' is ignored. Default: None
        :param scale_mode: {'cycle', 'iterations'}. Defines whether scale_fn is evaluated on cycle number or
                           cycle iterations (training iterations since start of cycle). Default: 'cycle'
        :param cycle_momentum: If ``True``, momentum is cycled inversely to learning rate between
                               'base_momentum' and 'max_momentum'. Default: True
        :param base_momentum: Initial momentum which is the lower boundary in the cycle for each parameter group.
                              Default: 0.8
        :param max_momentum: Upper momentum boundaries in the cycle for each parameter group.
                             Functionally, it defines the cycle amplitude (max_momentum - base_momentum).
                             The momentum at any cycle is the difference of max_momentum and some scaling of the amplitude;
                             therefore base_momentum may not actually be reached depending on scaling function.
                             Default: 0.9
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        """
        super().__init__(start_epoch, end_epoch, update_frequency=-1.0)
        self._lr_kwargs = {
            'base_lr': base_lr, 'max_lr': max_lr, 'step_size_up': step_size_up, 'step_size_down': step_size_down,
            'mode': mode, 'gamma': gamma, 'scale_fn': scale_fn, 'scale_mode': scale_mode,
            'cycle_momentum': cycle_momentum, 'base_momentum': base_momentum, 'max_momentum': max_momentum
        }
        self._lr_scheduler = None
        self._init_lr_set = False

    @property
    def lr_kwargs(self) -> Dict:
        return self._lr_kwargs

    @lr_kwargs.setter
    def lr_kwargs(self, value: Dict):
        if self._initialized:
            raise RuntimeError('Cannot change lr_kwargs after {} has been initialized'
                               .format(self.__class__.__name__))

        self._lr_kwargs = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Create the lr_scheduler using the optimizer and the provided lr_kwargs

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(CyclicLRModifier, self).initialize(module, optimizer)
        self._lr_scheduler = CyclicLR(optimizer=optimizer, **self._lr_kwargs)

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Calls into the lr scheduler to step for each batch

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        batch_count = round(epoch * steps_per_epoch)
        self._lr_scheduler.step(batch_count)


yaml.add_constructor(CyclicLRModifier.YAML_KEY, CyclicLRModifier.yaml_constructor)
yaml.add_constructor(CyclicLRModifier.YAML_KEY, CyclicLRModifier.yaml_constructor, yaml.SafeLoader)
