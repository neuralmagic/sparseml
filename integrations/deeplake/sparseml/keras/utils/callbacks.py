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
Built-in callbacks for Keras
"""

from typing import List, Union

from tensorflow import Tensor

from sparseml.keras.utils.compat import keras
from sparseml.keras.utils.logger import KerasLogger, LoggingMode


__all__ = [
    "LoggerSettingCallback",
    "LossesAndMetricsLoggingCallback",
]


class LoggerSettingCallback(keras.callbacks.Callback):
    """
    Class to help correctly set logging modes for callbacks that rely on KerasLogger.
    All callbacks using KerasLogger should derive from this class.

    :param loggers: logger or list of loggers
    """

    def __init__(self, loggers: Union[KerasLogger, List[KerasLogger]]):
        self._loggers = loggers if isinstance(loggers, list) else [loggers]

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the begin of a training epoch

        :param epoch: epoch index
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_epoch_begin(epoch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of a training epoch

        :param epoch: epoch index
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_epoch_end(epoch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Called at the begin of a batch in prediction

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_predict_batch_begin(batch, logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_predict_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in prediction

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_predict_batch_end(batch, logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_predict_begin(self, logs=None):
        """
        Called at the begin of prediction

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_predict_begin(logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_predict_end(self, logs=None):
        """
        Called at the end of prediction

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_predict_end(logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_test_batch_begin(self, batch, logs=None):
        """
        Called at the begin of a batch in evaluation

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_test_batch_begin(batch, logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_test_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in evaluation

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_test_batch_end(batch, logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_test_begin(self, logs=None):
        """
        Called at the begin of evaluation

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_test_begin(logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_test_end(self, logs=None):
        """
        Called at the end of evaluation

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_test_end(logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called at the begin of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_batch_begin(batch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_batch_end(batch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_train_begin(self, logs=None):
        """
        Called at the begin of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_begin(logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_train_end(self, logs=None):
        """
        Called at the end of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_end(logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def _set_logging_mode(self, mode: LoggingMode):
        for logger in self._loggers:
            logger.mode = mode


class LossesAndMetricsLoggingCallback(LoggerSettingCallback):
    """
    Callback to log all losses and metrics

    :param loggers: logger or list of loggers
    :param start_step: a start step tensor when this callback starts to take effect
    """

    def __init__(
        self,
        loggers: Union[KerasLogger, List[KerasLogger]],
        start_step: Union[Tensor, int] = 0,
    ):
        super().__init__(loggers)
        self._start_step = start_step
        self._step = None

    def on_train_begin(self, logs=None):
        """
        Called at the begin of training

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_begin(logs)
        self._step = keras.backend.get_value(self._start_step)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of a training epoch

        :param epoch: epoch index
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_epoch_end(epoch, logs)
        if logs is None:
            return
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TRAIN
            for tag, value in logs.items():
                logger.log_scalar("epoch_{}".format(tag), value, step=epoch)

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch in training

        :param batch: batch index in current epoch
        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_train_batch_end(batch, logs)
        if logs is None:
            return
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TRAIN
            if logger.update_freq == "batch" or (
                isinstance(logger.update_freq, int)
                and self._step % logger.update_freq == 0
            ):
                for tag, value in logs.items():
                    logger.log_scalar("batch_{}".format(tag), value, step=self._step)

        self._step += 1

    def on_test_end(self, logs=None):
        """
        Called at the end of evaluation

        :param logs: dictionary of logs (see Keras Callback doc)
        """
        super().on_test_end(logs)
        if logs is None:
            return
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TEST
            for tag, value in logs.items():
                logger.log_scalar("val_{}".format(tag), value, step=self._step)
