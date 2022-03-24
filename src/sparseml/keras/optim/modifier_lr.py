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
Learning rate modifiers for Keras models
"""

from typing import Dict, List, Union

from tensorflow import Tensor

from sparseml.keras.optim.modifier import (
    KerasModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.keras.utils import KerasLogger, LoggerSettingCallback, LoggingMode, keras
from sparseml.sparsification import LearningRateModifier as BaseLearningRateModifier
from sparseml.sparsification import (
    SetLearningRateModifier as BaseSetLearningRateModifier,
)


__all__ = ["SetLearningRateModifier", "LearningRateModifier"]


class LRModifierCallback(keras.callbacks.Callback):
    """
    Callback to modify learning rate of an optimizer

    :param optimizer: an optimizer whose lr needs updated
    :param start_step: start step when the lr needs to be updated
    :param end_step: end step of the update
    :param learning_rate: learning rate or learning rate schedule to be used
    """

    def __init__(
        self,
        optimizer: keras.optimizers.Optimizer,
        start_step: int,
        end_step: int,
        learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule],
    ):
        self._optimizer = optimizer
        self._start_step = start_step
        self._end_step = end_step
        self._learning_rate = learning_rate
        self._step = None

    def on_train_begin(self, logs=None):
        """
        Called at the begin of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        self._step = keras.backend.get_value(self._optimizer.iterations)

    def on_batch_begin(self, batch, logs=None):
        self.on_train_batch_begin(batch, logs=logs)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called at the begin of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        if self._step == self._start_step:
            setattr(self._optimizer, "lr", self._learning_rate)
        if self._step == self._end_step:
            assert self._end_step > -1
            persist_lr = self._optimizer.lr(self._step)
            setattr(self._optimizer, "lr", persist_lr)

    def on_batch_end(self, batch, logs=None):
        self.on_train_batch_end(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        self._step = self._step + 1


class LearningRateLoggingCallback(LoggerSettingCallback):
    """
    Callback to log the learning rate. No effect if global step is outside
    [start_step, end_step); otherwise the earnig rate is logged in the following cases:
    (1) at the end of an epoch;
    (2) at the right step if the update_freq attribute for batch logging is set in
        some logger;
    (3) the learning rate changes from previous logged value

    :param loggers: logger or a list of loggers
    :param start_step: starting step when the logging should start
    :param end_step: end step when the logging should stop
    """

    def __init__(
        self,
        loggers: Union[KerasLogger, List[KerasLogger]],
        start_step: int,
        end_step: int,
    ):
        super().__init__(loggers)
        self._prev_lr = None
        self._start_step = start_step
        self._end_step = end_step

    def on_train_begin(self, logs=None):
        """
        Called at the begin of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_begin(logs)
        self._step = keras.backend.get_value(self.model.optimizer.iterations)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the begin of a training epoch

        :param epoch: epoch index
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_epoch_begin(epoch, logs)
        if self._is_logging_step():
            lr_val = self._get_lr()
            for logger in self._loggers:
                assert logger.mode == LoggingMode.TRAIN
                if logger.update_freq == "epoch":
                    logger.log_scalar("learning_rate", lr_val, step=self._step)
                    self._prev_lr = lr_val

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called at the begin of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_batch_begin(batch, logs)
        if self._is_logging_step():
            lr_val = self._get_lr()
            for logger in self._loggers:
                assert logger.mode == LoggingMode.TRAIN
                should_log = (
                    logger.update_freq == "batch"
                    or (
                        isinstance(logger.update_freq, int)
                        and self._step % logger.update_freq == 0
                    )
                    or self._prev_lr is None
                    or self._prev_lr != lr_val
                )
                if should_log:
                    logger.log_scalar("learning_rate", lr_val, step=self._step)
                    self._prev_lr = lr_val

        self._step += 1

    def _get_lr(self):
        lr = self.model.optimizer.lr
        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            lr_val = lr(self.model.optimizer.iterations)
        else:
            lr_val = keras.backend.get_value(lr)
        return lr_val

    def _is_logging_step(self):
        return self._step >= self._start_step and (
            self._end_step == -1 or self._step < self._end_step
        )


@KerasModifierYAML()
class SetLearningRateModifier(BaseSetLearningRateModifier, ScheduledModifier):
    """
    Modifier to set the learning rate to a specific value at a certain point
    in the training process. Once that point is reached, will update the optimizer's
    params with the learning rate

    | Sample yaml:
    |    !SetLearningRateModifier
    |        start_epoch: 0.0
    |        learning_rate: 0.001

    :param learning_rate: The learning rate to use once this modifier starts
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: unused and should not be set
    """

    def __init__(
        self,
        learning_rate: float,
        start_epoch: float = -1,
        end_epoch: float = -1,
    ):
        super(SetLearningRateModifier, self).__init__(
            learning_rate=learning_rate,
            start_epoch=start_epoch,
            end_epoch=-1,
            end_comparator=None,
        )

    def _conditional_lr_update(self, epoch, current_lr):
        if epoch < self.start_epoch:
            return current_lr
        return self.learning_rate

    def modify(
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        loggers: Union[KerasLogger, List[KerasLogger]] = None,
        input_tensors: Tensor = None,
    ):
        """
        Modify model and optimizer, and provide callbacks to process the model

        :param model: a model to be modified with prunable layers wrapped by masks
        :param optimizer: an optimizer to be modified
        :param steps_per_epoch: number of steps per epoch
        :param loggers: list of loggers
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """
        model, optimizer, callback = super(SetLearningRateModifier, self).modify(
            model,
            optimizer,
            steps_per_epoch,
            loggers=loggers,
            input_tensors=input_tensors,
        )
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        assert end_step == -1
        lr_callback = LRModifierCallback(
            optimizer, start_step, end_step, self.learning_rate
        )
        lr_logging_callback = LearningRateLoggingCallback(loggers, start_step, end_step)
        return model, optimizer, [lr_callback, lr_logging_callback]


class _ExponentialDecay(keras.optimizers.schedules.ExponentialDecay):
    def __init__(
        self,
        start_step,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        name=None,
    ):
        super().__init__(
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase=staircase,
            name=name,
        )
        self._start_step = start_step

    @property
    def start_step(self):
        return self._start_step

    def __call__(self, step):
        if step < self.start_step:
            raise ValueError("Invalid step passed in")
        steps_count = step - self.start_step
        return super().__call__(steps_count)

    def get_config(self):
        config = super().get_config()
        config.update({"start_step": self.start_step})
        return config


class _PiecewiseConstantDecay(keras.optimizers.schedules.PiecewiseConstantDecay):
    def __init__(self, start_step, boundaries, values, name=None):
        super().__init__(boundaries, values, name=name)
        self._start_step = start_step

    @property
    def start_step(self):
        return self._start_step

    def __call__(self, step):
        if step < self.start_step:
            raise ValueError("Invalid step passed in")
        steps_count = step - self.start_step
        return super().__call__(steps_count)

    def get_config(self):
        config = super().get_config()
        config.update({"start_step": self.start_step})
        return config


@KerasModifierYAML()
class LearningRateModifier(BaseLearningRateModifier, ScheduledUpdateModifier):
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
    """

    def __init__(
        self,
        lr_class: str,
        lr_kwargs: Dict,
        init_lr: float,
        start_epoch: float,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
    ):
        super(LearningRateModifier, self).__init__(
            lr_class=lr_class,
            lr_kwargs=lr_kwargs,
            init_lr=init_lr,
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
            learning_rate = _ExponentialDecay(
                start_step,
                self.init_lr,
                lr_kwargs["step_size"],
                lr_kwargs["gamma"],
                staircase=True,
                name="StepLR",
            )
        elif lr_class == "MultiStepLR":
            boundaries = lr_kwargs["milestones"]
            values = [
                self.init_lr * (lr_kwargs["gamma"] ** k)
                for k in range(len(boundaries) + 1)
            ]
            learning_rate = _PiecewiseConstantDecay(
                start_step, boundaries, values, name="MultiStepLR"
            )
        elif lr_class == "ExponentialLR":
            learning_rate = _ExponentialDecay(
                start_step,
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
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        loggers: Union[KerasLogger, List[KerasLogger]] = None,
        input_tensors: Tensor = None,
    ):
        """
        Modify model and optimizer, and provide callbacks to process the model

        :param model: a model to be modified with prunable layers wrapped by masks
        :param optimizer: an optimizer to be modified
        :param steps_per_epoch: number of steps per epoch
        :param loggers: list of loggers
        :param input_tensors: optional input tensors
        :return: modified model, optimizer and callbacks
        """

        model, optimizer, callback = super(LearningRateModifier, self).modify(
            model,
            optimizer,
            steps_per_epoch,
            loggers=loggers,
            input_tensors=input_tensors,
        )
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        learning_rate = self._create_learning_rate_scheduler(steps_per_epoch)
        lr_callback = LRModifierCallback(optimizer, start_step, end_step, learning_rate)
        lr_logging_callback = LearningRateLoggingCallback(loggers, start_step, end_step)
        return model, optimizer, [lr_callback, lr_logging_callback]
