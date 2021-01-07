"""
Learning rate modifiers for Keras models
"""
from copy import deepcopy
from typing import Dict, List, Union

import tensorflow as tf
from sparseml.keras.optim.modifier import (
    KerasModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.optim import LearningRate, SetLearningRate
from sparseml.utils import ALL_TOKEN


__all__ = ["SetLearningRateModifier", "LearningRateModifier"]


class LRModifierCallback(tf.keras.callbacks.Callback):
    """
    Callback to modify learning rate of an optimizer

    :param optimizer: an optimizer whose lr needs updated
    :param start_step: start step when the lr needs to be updated
    :param end_step: end step of the update
    :param learning_rate: learning rate or learning rate schedule to be used
    """

    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        start_step: int,
        end_step: int,
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
    ):
        self.optimizer = optimizer
        self.prev_lr = deepcopy(self.optimizer.lr)
        self.start_step = start_step
        self.end_step = end_step
        self.learning_rate = learning_rate
        self.step = None

    def on_train_begin(self, logs=None):
        self.step = tf.keras.backend.get_value(self.model.optimizer.iterations)

    def on_train_batch_begin(self, batch, logs=None):
        if self.step == self.start_step:
            tf.keras.backend.set_value(self.optimizer.lr, self.learning_rate)
        if self.step == self.end_step:
            assert self.end_step > -1
            persist_lr = self.optimizer.lr(self.step)
            tf.keras.backend.set_value(self.optimizer.lr, persist_lr)

    def on_train_batch_end(self, batch, logs=None):
        self.step = self.step + 1


@KerasModifierYAML()
class SetLearningRateModifier(ScheduledModifier, SetLearningRate):
    """
    Modifier to set the learning rate to a specific value at a certain point
    in the training process. Once that point is reached, will update the optimizer's
    params with the learning rate

    | Sample yaml:
    |    !SetLearningRateModifier
    |        start_epoch: 0.0
    |        learning_rate: 0.001
    |        log_types: __ALL__

    :param learning_rate: The learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: unused and should not be set
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(
        self,
        learning_rate: float,
        start_epoch: float = -1,
        end_epoch: float = -1,
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super().__init__(
            learning_rate=learning_rate,
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=None,
        )

    def _conditional_lr_update(self, epoch, current_lr):
        if epoch < self.start_epoch:
            return current_lr
        return self.learning_rate

    def modify(
        self, model, optimizer, steps_per_epoch: int, input_tensors: tf.Tensor = None
    ):
        """
        Modify model and optimizer, and provide callbacks to process the model

        :param model: a model to be modified with prunable layers wrapped by masks
        :param optimizer: an optimizer to be modified
        :param steps_per_epoch: number of steps per epoch
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """
        model, optimizer, callback = super(SetLearningRateModifier, self).modify(
            model, optimizer, steps_per_epoch, input_tensors=input_tensors
        )
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        assert end_step == -1
        lr_callback = LRModifierCallback(
            optimizer, start_step, end_step, self.learning_rate
        )
        return model, optimizer, lr_callback


@KerasModifierYAML()
class LearningRateModifier(ScheduledUpdateModifier, LearningRate):
    """
    Modifier to set the learning rate to follow specific schedulers
    within a period of epochs.
    The following schedulers are current supported: ExponentialLR,
    StepLR, MultiStepLR

    | Sample yaml:
    |    !LearningRateModifier
    |        lr_class: ExponentialDecay
    |        lr_kwargs:
    |            initial_learning_rate: 0.01
    |            decay_steps: 10000
    |            decay_rate: 0.96
    |        start_epoch: 0.0
    |        end_epoch: 10.0
    |        log_types: __ALL__

    :param lr_class: The name of the lr scheduler class to use:
        [StepLR, MultiStepLR, ExponentialLR]
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
    ):
        super().__init__(
            lr_class=lr_class,
            lr_kwargs=lr_kwargs,
            init_lr=init_lr,
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=-1,
            end_comparator=-1,
        )

    def _create_learning_rate_scheduler(self, steps_per_epoch):
        lr_class, lr_kwargs = self.corrected_lr_info(
            steps_per_epoch, self.start_epoch, self.end_epoch
        )
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)

        if lr_class == "StepLR":
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                self.init_lr,
                lr_kwargs["step_size"],
                lr_kwargs["gamma"],
                staircase=True,
                name="StepLR",
            )
        elif lr_class == "MultiStepLR":
            boundaries = lr_kwargs["milestones"]
            values = [
                self.init_lr * (lr_kwargs["gamma"] ^ k) for k in range(len(boundaries))
            ]
            learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values, name="MultiStepLR"
            )
        elif lr_class == "ExponentialLR":
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                self.init_lr,
                lr_kwargs["step_size"],
                lr_kwargs["gamma"],
                staircase=False,
                name="ExponentialLR",
            )
        else:
            raise ValueError("unrecognized lr_class given of {}".format(lr_class))
        return learning_rate

    def modify(
        self, model, optimizer, steps_per_epoch: int, input_tensors: tf.Tensor = None
    ):
        """
        Modify model and optimizer, and provide callbacks to process the model

        :param model: a model to be modified with prunable layers wrapped by masks
        :param optimizer: an optimizer to be modified
        :param steps_per_epoch: number of steps per epoch
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """

        model, optimizer, callback = super(LearningRateModifier, self).modify(
            model, optimizer, steps_per_epoch, input_tensors=input_tensors
        )
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        learning_rate = self._create_learning_rate_scheduler(steps_per_epoch)
        lr_callback = LRModifierCallback(optimizer, start_step, end_step, learning_rate)
        return model, optimizer, lr_callback
