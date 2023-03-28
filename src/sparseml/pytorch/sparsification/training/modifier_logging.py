from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.utils import log_module_sparsification_info
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)

__all__ = ["SparsificationLoggingModifier"]


@PyTorchModifierYAML()
class SparsificationLoggingModifier(ScheduledUpdateModifier):
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
        start_epoch: float,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        end_comparator: float = -1
    ):
        super(SparsificationLoggingModifier, self).__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            end_comparator=end_comparator,
        )

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
        log_module_sparsification_info(module=module, logger=self.loggers)

