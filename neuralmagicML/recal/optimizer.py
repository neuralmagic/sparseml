from typing import List, Union
import sys
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .manager import ScheduledModifierManager
from .logger import ModifierLogger


__all__ = ["ScheduledOptimizer"]


class ScheduledOptimizer(Optimizer):
    """
    An optimizer wrapper to handle applying modifiers according to their schedule to both
    the passed in optimizer and the module

    Overrides the step() function so that this method can call before and after on the modifiers
    to apply appropriate modifications to both the optimizer and the module

    Required to either pass in steps_per_epoch or to call epoch_start() and epoch_end()
    Doing both is recommended as there are some caveats when only one is used

    Only using steps_per_epoch:
    using this we estimate when epoch_start and epoch_end are based on how many steps we take
    can result in inaccurate estimates of starting a new epoch:
      - varying batch sizes
      - changes in dataset size
      - irregular optimization routines

    Only using epoch_start() and epoch_end():
    based on this we estimate the steps_per_epoch to give a finer granularity of control after first epoch
    can result in inaccurate estimates of the current batch within epoch:
      - info not available until first epoch complete (one cycle of epoch_start() and epoch_end())
      - changes in dataset size
      - irregular optimization routines

    Lifecycle:
    - epoch_start
    - training
        - zero_grad
        - loss_update
            - modifiers.loss_update
        - step
            - modifiers.update
            - modifiers.optimizer_pre_step
            - optimizer.step
            - modifiers.optimizers_post_step
    - epoch_end
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: Module,
        manager: Union[ScheduledModifierManager, List[ScheduledModifierManager]],
        steps_per_epoch: int,
        loggers: Union[List[ModifierLogger], None] = None,
    ):
        """
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param manager: the manager or list of managers used to apply modifications
        :param steps_per_epoch: the number of steps or batches in each epoch, not strictly required and can be set to -1
                                used to calculate decimals within the epoch, when not using can result in irregularities
        :param loggers: loggers to log important info to within the modifiers; ex tensorboard or to the console
        """
        # do not call into super.__init__()
        # makes the implementation messier activation this instance is not actually acting activation an optimizer
        # just a wrapper around the passed in optimizer
        self._optimizer = optimizer
        self._module = module
        self._managers = (
            [manager] if isinstance(manager, ScheduledModifierManager) else manager
        )
        self._steps_per_epoch = steps_per_epoch

        self._steps = 0
        self._epoch_counter = -1
        self._epoch = -1
        self._epoch_steps = 0
        self._epoch_started = False
        self._mode = None

        for manager in self._managers:
            manager.initialize(self._module, self._optimizer)
            manager.initialize_loggers(loggers)

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def __setstate__(self, state):
        self._optimizer.__setstate__(state)

    def __repr__(self):
        self._optimizer.__repr__()

    @property
    def learning_rate(self) -> float:
        """
        :return: convenience function to get the first learning rate for any of the param groups in the optimizer
        """
        for param_group in self.param_groups:
            return param_group["lr"]

        raise RuntimeError("cannot get learning_rate, no param_groups available")

    @learning_rate.setter
    def learning_rate(self, value: float):
        """
        :param value: the learning rate to set for the optimizer, will set all param groups in the optim to this value
        """
        for param_group in self.param_groups:
            param_group["lr"] = value

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value

    @property
    def min_epochs(self) -> int:
        """
        :return: the minimum epochs required by any of the modifiers under any of the manager(s)
        """
        vals = [man.min_epochs for man in self._managers]

        return min(vals) if len(vals) > 0 else -1

    @property
    def max_epochs(self) -> int:
        """
        :return: the maximum number of epochs required by any of the modifiers under any of the manager(s)
        """
        vals = [man.min_epochs for man in self._managers]

        return max(vals) if len(vals) > 0 else -1

    def state_dict(self):
        self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self, closure=None):
        """
        Called to perform a step on the optimizer activation normal
        Updates the current epoch based on the step count
        Calls into modifiers before the step happens
        Calls into modifiers after the step happens

        :param closure: optional closure passed into the contained optimizer for the step
        """
        if self._mode is None and self._steps_per_epoch <= 0:
            raise RuntimeError(
                "epoch_start must be called or steps_per_epoch must be supplied in the constructor"
            )

        if self._mode is None:
            self._mode = "counter"

        if self._mode == "counter":
            self._set_epoch_by_counter()
        elif self._mode == "start_end":
            self._set_epoch_by_start_end()
        else:
            raise ValueError("unknown mode of {}".format(self._mode))

        self._epoch_steps += 1

        for manager in self._managers:
            manager.update(
                self._module, self._optimizer, self._epoch, self._steps_per_epoch
            )

        for manager in self._managers:
            manager.optimizer_pre_step(
                self._module, self._optimizer, self._epoch, self._steps_per_epoch
            )

        self._optimizer.step(closure)

        for manager in self._managers:
            manager.optimizer_post_step(
                self._module, self._optimizer, self._epoch, self._steps_per_epoch
            )

    def add_param_group(self, param_group):
        self._optimizer.add_param_group(param_group)

    def epoch_start(self):
        """
        Called before starting an epoch for training

        Calls into the managers to update based on the new epoch that is starting
        """
        if self._mode is not None and self._mode != "start_end":
            raise RuntimeError(
                "first epoch_start call must happen before first step call"
            )

        self._mode = "start_end"
        self._epoch_started = True
        self._epoch_counter += 1
        self._epoch = float(self._epoch_counter)
        self._epoch_steps = 0

        for manager in self._managers:
            manager.update(
                self._module, self._optimizer, self._epoch, self._steps_per_epoch
            )

    def epoch_end(self):
        """
        Called after an epoch for training has ended

        Calls into the managers to update based on the epoch that just ended
        """
        if self._mode != "start_end":
            raise RuntimeError("epoch_start call must happen before epoch_end")

        self._epoch = self._epoch_counter - sys.float_info.epsilon

        for manager in self._managers:
            manager.update(
                self._module, self._optimizer, self._epoch, self._steps_per_epoch
            )

    def loss_update(self, loss: Tensor) -> Tensor:
        """
        Optional call to update modifiers based on the calculated loss
        Not needed unless one or more of the modifier is using the loss to make a modification
        or is modifying the loss itself

        :param loss: the calculated loss after running a forward pass and loss_fn
        :return: the modified loss tensor
        """
        for manager in self._managers:
            loss = manager.loss_update(
                loss, self._module, self._optimizer, self._epoch, self._steps_per_epoch
            )

        return loss

    def _set_epoch_by_counter(self):
        if self._epoch_steps >= self._steps_per_epoch or self._epoch_counter < 0:
            self._epoch_counter += 1
            self._epoch_steps = 0

        self._epoch = float(self._epoch_counter) + float(self._epoch_steps) / float(
            self._steps_per_epoch
        )
        self._check_epoch_bounds()

    def _set_epoch_by_start_end(self):
        if self._steps_per_epoch <= 0:
            self._epoch = float(self._epoch_counter)
        else:
            self._epoch = float(self._epoch_counter) + float(self._epoch_steps) / float(
                self._steps_per_epoch
            )

        self._check_epoch_bounds()

    def _check_epoch_bounds(self):
        if self._epoch >= self._epoch_counter + 1:
            self._epoch = float(self._epoch_counter + 1) - sys.float_info.epsilon
