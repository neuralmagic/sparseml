"""
Contains code for loggers that help visualize the information from each modifier
"""

import logging
import time
from abc import ABC, abstractmethod
from logging import Logger
from typing import Union

import tensorflow
from tensorflow.summary import create_file_writer


__all__ = ["KerasLogger", "PythonLogger", "TensorBoardLogger"]


class KerasLogger(ABC):
    """
    Base class that all modifier loggers must implement.

    :param name: name given to the logger, used for identification
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """
        :return: name given to the logger, used for identification
        """
        return self._name

    @abstractmethod
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Union[None, int] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        """
        raise NotImplementedError()


class PythonLogger(KerasLogger):
    """
    Modifier logger that handles printing values into a python logger instance.

    :param logger: a logger instance to log to, if None then will create it's own
    :param name: name given to the logger, used for identification;
        defaults to python
    """

    def __init__(
        self,
        logger: Logger = None,
        update_freq: Union[str, int] = "epoch",
        name: str = "python",
    ):
        super().__init__(name)

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
        self._update_freq = update_freq

    @property
    def update_freq(self):
        return self._update_freq

    def __getattr__(self, item):
        return getattr(self._logger, item)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken,
            defaults to time.time()
        """
        if wall_time is None:
            wall_time = time.time()

        msg = "{}-SCALAR {} [{} - {}]: {}".format(
            self.name, tag, step, wall_time, value
        )
        self._logger.info(msg)


class TensorBoardLogger(KerasLogger):
    """
    Modifier logger that handles outputting values into a TensorBoard log directory
    for viewing in TensorBoard.

    :param log_dir: the path to create a SummaryWriter at. writer must be None
        to use if not supplied (and writer is None),
        will create a TensorBoard dir in cwd
    :param name: name given to the logger, used for identification;
        defaults to tensorboard
    """

    def __init__(
        self,
        log_dir: str = "logs",
        update_freq: Union[str, int] = "epoch",
        name: str = "tensorboard",
        **kwargs,
    ):
        super().__init__(name)
        self._update_freq = update_freq
        self._writer = create_file_writer(log_dir)

    @property
    def update_freq(self):
        return self._update_freq

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Union[None, int] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        """
        with self._writer.as_default():
            tensorflow.summary.scalar(tag, data=value, step=step)
            self._writer.flush()
