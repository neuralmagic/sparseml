from typing import List, Union
import sys
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .modifier import ScheduledModifierManager


__all__ = ['ScheduledOptimizer']


class ScheduledOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, module: Module,
                 manager: Union[ScheduledModifierManager, List[ScheduledModifierManager]], steps_per_epoch: int):
        """
        An optimizer wrapper to handle applying modifiers according to their schedule to both
        the passed in optimizer and the module

        Overrides the step() function so that this method can call before and after on the modifiers
        to apply appropriate modifications to both the optimizer and the module

        Required to either pass in steps_per_epoch or to call epoch_start() and epoch_end()
        Doing both results in best use cases as there are some caveats when only one is used

        Only using steps_per_epoch:
        using this we estimate when epoch_start and epoch_end are based on how many steps we take
        can result in inaccurate estimates of starting a new epoch:
          - First batch will not be modified until optimizer.step()
          - varying batch sizes
          - changes in dataset size
          - irregular optimization routines

        Only using epoch_start() and epoch_end():
        based on this we estimate the steps_per_epoch to give a finer granularity of control after first epoch
        can result in inaccurate estimates of the current batch within epoch:
          - info not available until first epoch complete (one cycle of epoch_start() and epoch_end())
          - changes in dataset size
          - irregular optimization routines

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param manager: the manager or list of managers used to apply modifications
        :param steps_per_epoch: the number of steps or batches in each epoch, not strictly required and can be set to -1
                                used to calculate decimals within the epoch, when not using can result in irregularities
        """
        # do not call into super.__init__()
        # makes the implementation messier as this instance is not actually acting as an optimizer
        # just a wrapper around the passed in optimizer
        self._optimizer = optimizer
        self._module = module
        self._managers = [manager] if isinstance(manager, ScheduledModifierManager) else manager
        self._steps = 0
        self._steps_per_epoch = steps_per_epoch
        self._epoch_counter = -1
        self._epoch = -1
        self._epoch_steps = 0
        self._epoch_started = False

        for manager in self._managers:
            manager.initialize(self._module, self._optimizer)

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def __setstate__(self, state):
        self._optimizer.__setstate__(state)

    def __repr__(self):
        self._optimizer.__repr__()

    @property
    def learning_rate(self) -> float:
        for param_group in self.param_groups:
            return param_group['lr']

        raise Exception('cannot get learning_rate, no param_groups available')

    @learning_rate.setter
    def learning_rate(self, value: float):
        for param_group in self.param_groups:
            param_group['lr'] = value

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value

    def state_dict(self):
        self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self, closure=None):
        """
        Called to perform a step on the optimizer as normal
        Updates the current epoch based on the step count
        Calls into modifiers before the step happens
        Calls into modifiers after the step happens

        :param closure: optional closure passed into the contained optimizer for the step
        """
        self._epoch_steps += 1
        self._epoch = self._calc_current_epoch()

        for manager in self._managers:
            manager.update(self._module, self._optimizer, self._epoch, self._steps_per_epoch)

        for manager in self._managers:
            manager.optimizer_pre_step(self._module, self._optimizer, self._epoch, self._steps_per_epoch)

        self._optimizer.step(closure)

        for manager in self._managers:
            manager.optimizer_post_step(self._module, self._optimizer, self._epoch, self._steps_per_epoch)

    def add_param_group(self, param_group):
        self._optimizer.add_param_group(param_group)

    def epoch_start(self):
        """
        Called before starting an epoch for training

        Calls into the managers to update based on the new epoch that is starting
        """
        self._epoch_started = True
        self._epoch_counter += 1
        self._epoch = float(self._epoch_counter)
        self._epoch_steps = 0

        for manager in self._managers:
            manager.update(self._module, self._optimizer, self._epoch, self._steps_per_epoch)

    def epoch_end(self):
        """
        Called after an epoch for training has ended

        Calls into the managers to update based on the epoch that just ended
        """
        self._epoch = self._epoch_counter - sys.float_info.epsilon

        for manager in self._managers:
            manager.update(self._module, self._optimizer, self._epoch, self._steps_per_epoch)

    def loss_update(self, loss: Tensor):
        """
        Optional call to update modifiers based on the calculated loss
        Not needed unless one or more of the modifier is using the loss to make a modification

        :param loss: the calculated loss after running a forward pass and loss_fn
        """
        for manager in self._managers:
            manager.loss_update(loss, self._module, self._optimizer, self._epoch, self._steps_per_epoch)

    def _calc_current_epoch(self):
        if self._steps_per_epoch <= 0:
            # steps per epoch are not provided, must work at an epoch granularity
            # required that epoch_start and epoch_end are called
            if not self._epoch_started:
                raise Exception('steps_per_epoch is not supplied for ScheduledOptimizer, '
                                'epoch_start and epoch_end must be called then')

            return float(self._epoch_counter)

        if not self._epoch_started and self._epoch_steps > self._steps_per_epoch:
            # epoch start not being called, must check for when we need to increment our epoch count
            self._epoch_counter += 1
            self._epoch = float(self._epoch_counter)
            self._epoch_steps = 0

        epoch = float(self._epoch_counter) + float(self._epoch_steps) / float(self._steps_per_epoch)

        if epoch >= self._epoch_counter + 1:
            epoch = float(self._epoch_counter + 1) - sys.float_info.epsilon

        return epoch
