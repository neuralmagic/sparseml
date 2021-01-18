"""
Built-in callbacks for Keras
"""

from typing import List, Union
import tensorflow
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from sparseml.keras.utils import KerasLogger, LoggingMode


__all__ = [
    "LoggerSettingCallback",
    "LossesAndMetricsLoggingCallback",
    "LearningRateLoggingCallback",
]


class LoggerSettingCallback(keras.callbacks.Callback):
    """
    Class to help correctly set logging modes for callbacks that rely on KerasLogger.
    All callbacks using KerasLogger should derive from this class.

    :param loggers: logger or list of loggers
    """

    def __init__(self, loggers: Union[KerasLogger, List[KerasLogger]]):
        self._loggers = loggers if isinstance(loggers, list) else [loggers]

    def _set_logging_mode(self, mode: LoggingMode):
        for logger in self._loggers:
            logger.mode = mode

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_predict_batch_begin(self, batch, logs=None):
        super().on_predict_batch_begin(batch, logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_predict_batch_end(self, batch, logs=None):
        super().on_predict_batch_end(batch, logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_predict_begin(self, logs=None):
        super().on_predict_begin(logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_predict_end(self, logs=None):
        super().on_predict_end(logs)
        self._set_logging_mode(LoggingMode.PREDICT)

    def on_test_batch_begin(self, batch, logs=None):
        super().on_test_batch_begin(batch, logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_test_batch_end(self, batch, logs=None):
        super().on_test_batch_end(batch, logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_test_begin(self, logs=None):
        super().on_test_begin(logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_test_end(self, logs=None):
        super().on_test_end(logs)
        self._set_logging_mode(LoggingMode.TEST)

    def on_train_batch_begin(self, batch, logs=None):
        super().on_train_batch_begin(batch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self._set_logging_mode(LoggingMode.TRAIN)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self._set_logging_mode(LoggingMode.TRAIN)


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
        super().on_train_begin(logs)
        self._step = tensorflow.keras.backend.get_value(self._start_step)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if logs is None:
            return
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TRAIN
            for tag, value in logs.items():
                logger.log_scalar("epoch_{}".format(tag), value, step=epoch)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if logs is None:
            return
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TRAIN
            if logger.update_freq == "batch" or self._step % logger.update_freq == 0:
                for tag, value in logs.items():
                    logger.log_scalar("batch_{}".format(tag), value, step=self._step)

        self._step += 1

    def on_test_end(self, logs=None):
        super().on_test_end(logs)
        if logs is None:
            return
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TEST
            for tag, value in logs.items():
                logger.log_scalar("val_{}".format(tag), value, step=self._step)


class LearningRateLoggingCallback(LoggerSettingCallback):
    """
    Callback to log the learning rate. Learnig rate is logged in the following cases:
    (1) at the end of an epoch;
    (2) at the right step if the update_freq attribute for batch logging is set in
        some logger;
    (3) the learning rate changes from previous logged value

    :param loggers: logger or a list of loggers
    """

    def __init__(self, loggers):
        super().__init__(loggers)
        self._prev_lr = None

    def _get_lr(self):
        lr = self.model.optimizer.lr
        if isinstance(lr, LearningRateSchedule):
            lr_val = lr(self.model.optimizer.iterations)
        else:
            lr_val = K.get_value(lr)
        return lr_val

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self._step = K.get_value(self.model.optimizer.iterations)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        lr_val = self._get_lr()
        for logger in self._loggers:
            assert logger.mode == LoggingMode.TRAIN
            if logger.update_freq == "epoch":
                logger.log_scalar("learning_rate", lr_val, step=self._step)
                self._prev_lr = lr_val

    def on_train_batch_begin(self, batch, logs=None):
        super().on_train_batch_begin(batch, logs)
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
