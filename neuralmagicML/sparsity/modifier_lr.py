from typing import Dict, Union
import yaml
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts
)

from .modifier import ScheduledUpdateModifier


__all__ = ['LearningRateModifier']


CONSTRUCTORS = {
    'StepLR': StepLR,
    'MultiStepLR': MultiStepLR,
    'ExponentialLR': ExponentialLR,
    'ReduceLROnPlateau': ReduceLROnPlateau,
    'CyclicLR': CyclicLR,
    'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts
}


class LearningRateModifier(ScheduledUpdateModifier):
    YAML_KEY = u'!LearningRateModifier'

    @staticmethod
    def yaml_constructor(loader, node):
        instance = LearningRateModifier.__new__(LearningRateModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(self, lr_class: str, lr_kwargs: Dict, init_lr: Union[float, None] = None,
                 start_epoch: float = -1.0, end_epoch: float = -1.0, update_frequency: float = -1.0):
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
        :param init_lr: The initial learning rate to use once this modifier starts
        :param start_epoch: The epoch to start the modifier at (set to -1.0 so it starts immediately)
        :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends)
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        """
        super().__init__(start_epoch, end_epoch, update_frequency)
        self._lr_class = lr_class
        self._lr_kwargs = lr_kwargs
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
            raise Exception('Cannot change lr_class after {} has been initialized'.format(self.__class__.__name__))

        self._lr_class = value

    @property
    def lr_kwargs(self) -> Dict:
        return self._lr_kwargs

    @lr_kwargs.setter
    def lr_kwargs(self, value: Dict):
        if self._initialized:
            raise Exception('Cannot change lr_kwargs after {} has been initialized'.format(self.__class__.__name__))

        self._lr_kwargs = value

    @property
    def init_lr(self) -> Union[float, None]:
        return self._init_lr

    @init_lr.setter
    def init_lr(self, value: Union[float, None]):
        if self._initialized:
            raise Exception('Cannot change init_lr after {} has been initialized'.format(self.__class__.__name__))

        self._init_lr = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Create the lr_scheduler using the optimizer and the provided lr_kwargs

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(LearningRateModifier, self).initialize(module, optimizer)
        self._lr_scheduler = CONSTRUCTORS[self._lr_class](optimizer=optimizer, **self._lr_kwargs)

    def update(self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int):
        """
        Calls into the lr scheduler to step given the epoch
        Additionally will first set the lr to the init_lr if not set yet

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """

        if not self._init_lr_set and self._init_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self._init_lr

            self._init_lr_set = True

        self._lr_scheduler.step(epoch)


yaml.add_constructor(LearningRateModifier.YAML_KEY, LearningRateModifier.yaml_constructor)
yaml.add_constructor(LearningRateModifier.YAML_KEY, LearningRateModifier.yaml_constructor, yaml.SafeLoader)
