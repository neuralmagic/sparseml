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
Contains code for loggers that help visualize the information from each modifier
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from enum import Enum, unique
from logging import Logger
from typing import Union

import tensorflow
from tensorflow.summary import create_file_writer


__all__ = ["KerasLogger", "PythonLogger", "TensorBoardLogger", "LoggingMode"]


@unique
class LoggingMode(Enum):
    """
    Some logger changes its logging behavior (e.g. the destination it logs to)
    based on whether it's running in train, validation or predict mode

    This enum defines the mode a logger could be in during its lifetime
    """

    TRAIN = "train"
    TEST = "validation"
    PREDICT = "predict"


class KerasLogger(ABC):
    """
    Base class that all modifier loggers shall implement.

    :param name: name given to the logger, used for identification
    :param update_freq: define when logging should happen
        - "epoch": log at the end of each epoch
        - "batch": log at the end of each training batch
        - an integer value: the number of batches before the next logging should be
    """

    def __init__(self, name: str, update_freq: Union[str, int] = "epoch", **kwargs):
        self._name = name
        self._update_freq = update_freq
        self._mode = LoggingMode.TRAIN

    @property
    def name(self) -> str:
        """
        :return: name given to the logger, used for identification
        """
        return self._name

    @property
    def update_freq(self):
        return self._update_freq

    @property
    def mode(self):
        """
        Mode the current logger is at a current time

        :return: a LoggingMode
        """
        return self._mode

    @mode.setter
    def mode(self, value: LoggingMode):
        """
        Set logging mode
        """
        if not isinstance(value, LoggingMode):
            raise ValueError("Expected LoggingMode for mode, got {}".format(value))
        self._mode = value

    @abstractmethod
    def log_scalar(
        self, tag: str, value: float, step: Union[None, int] = None, **kwargs
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

    :param name: name given to the logger, used for identification;
    :param update_freq: update frequency (see attribute info in the base class)
    :param logger: a logger instance to log to, if None then will create it's own
        defaults to python
    """

    def __init__(
        self,
        name: str = "python",
        update_freq: Union[str, int] = "epoch",
        logger: Logger = None,
    ):
        super().__init__(name, update_freq=update_freq)
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
        self._update_freq = update_freq

    def __getattr__(self, item):
        return getattr(self._logger, item)

    def log_scalar(
        self, tag: str, value: float, step: Union[None, int] = None, **kwargs
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken,
            defaults to time.time()
        """
        wall_time = kwargs.get("wall_time")
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

    :param name: name given to the logger, used for identification;
        defaults to tensorboard
    :param update_freq: update frequency (see attribute info in the base class)
    :param log_dir: the path to create a SummaryWriter at. writer must be None
        to use if not supplied (and writer is None),
        will create a TensorBoard dir in cwd
    """

    def __init__(
        self,
        name: str = "tensorboard",
        update_freq: Union[str, int] = "epoch",
        log_dir: str = "logs",
    ):
        super().__init__(name, update_freq=update_freq)
        self._writers = {}  # Lazy initialization
        self._log_dir = log_dir

    def _get_active_writer(self):
        if self.mode not in self._writers:
            self._writers[self.mode] = create_file_writer(
                os.path.join(self._log_dir, self.mode.value)
            )
        return self._writers[self.mode]

    def log_scalar(
        self, tag: str, value: float, step: Union[int, None] = None, **kwargs
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        """
        description = kwargs.get("description")
        writer = self._get_active_writer()
        with writer.as_default():
            tensorflow.summary.scalar(
                tag, data=value, step=step, description=description
            )
            writer.flush()
