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
from logging import Logger
from typing import Dict, Union

from numpy import ndarray
from torch import Tensor


try:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ModuleNotFoundError, ImportError):
        from tensorboardX import SummaryWriter
    tensorboard_import_error = None
except Exception as tensorboard_err:
    SummaryWriter = None
    tensorboard_import_error = tensorboard_err

from sparseml.utils import create_dirs


__all__ = ["PyTorchLogger", "PythonLogger", "TensorBoardLogger"]


class PyTorchLogger(ABC):
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
    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        raise NotImplementedError()

    @abstractmethod
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
        :param wall_time: global wall time for when the value was taken
        """
        raise NotImplementedError()

    @abstractmethod
    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        """
        raise NotImplementedError()

    @abstractmethod
    def log_histogram(
        self,
        tag: str,
        values: Union[Tensor, ndarray],
        bins: str = "tensorflow",
        max_bins: Union[int, None] = None,
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the histogram with
        :param values: values to log as a histogram
        :param bins: the type of bins to use for grouping the values,
            follows tensorboard terminology
        :param max_bins: maximum number of bins to use (default None)
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        """
        raise NotImplementedError()

    @abstractmethod
    def log_histogram_raw(
        self,
        tag: str,
        min_val: Union[float, int],
        max_val: Union[float, int],
        num_vals: int,
        sum_vals: Union[float, int],
        sum_squares: Union[float, int],
        bucket_limits: Union[Tensor, ndarray],
        bucket_counts: Union[Tensor, ndarray],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the histogram with
        :param min_val: min value
        :param max_val: max value
        :param num_vals: number of values
        :param sum_vals: sum of all the values
        :param sum_squares: sum of the squares of all the values
        :param bucket_limits: upper value per bucket
        :param bucket_counts: number of values per bucket
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        """
        raise NotImplementedError()


class PythonLogger(PyTorchLogger):
    """
    Modifier logger that handles printing values into a python logger instance.

    :param logger: a logger instance to log to, if None then will create it's own
    :param name: name given to the logger, used for identification;
        defaults to python
    """

    def __init__(self, logger: Logger = None, name: str = "python"):
        super().__init__(name)

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

    def __getattr__(self, item):
        return getattr(self._logger, item)

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        msg = "{}-HYPERPARAMS:\n".format(self.name) + "\n".join(
            "   {}: {}".format(key, value) for key, value in params.items()
        )
        self._logger.info(msg)

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

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        if wall_time is None:
            wall_time = time.time()

        msg = "{}-SCALARS {} [{} - {}]:\n".format(
            self.name, tag, step, wall_time
        ) + "\n".join("{}: {}".format(key, value) for key, value in values.items())
        self._logger.info(msg)

    def log_histogram(
        self,
        tag: str,
        values: Union[Tensor, ndarray],
        bins: str = "tensorflow",
        max_bins: Union[int, None] = None,
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the histogram with
        :param values: values to log as a histogram
        :param bins: the type of bins to use for grouping the values,
            follows tensorboard terminology
        :param max_bins: maximum number of bins to use (default None)
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        if wall_time is None:
            wall_time = time.time()

        msg = "{}-HISTOGRAM {} [{} - {}]: cannot log".format(
            self.name, tag, step, wall_time
        )
        self._logger.info(msg)

    def log_histogram_raw(
        self,
        tag: str,
        min_val: Union[float, int],
        max_val: Union[float, int],
        num_vals: int,
        sum_vals: Union[float, int],
        sum_squares: Union[float, int],
        bucket_limits: Union[Tensor, ndarray],
        bucket_counts: Union[Tensor, ndarray],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the histogram with
        :param min_val: min value
        :param max_val: max value
        :param num_vals: number of values
        :param sum_vals: sum of all the values
        :param sum_squares: sum of the squares of all the values
        :param bucket_limits: upper value per bucket
        :param bucket_counts: number of values per bucket
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        if wall_time is None:
            wall_time = time.time()

        msg = "{}-HISTOGRAM {} [{} - {}]: cannot log".format(
            self.name, tag, step, wall_time
        )
        self._logger.info(msg)


class TensorBoardLogger(PyTorchLogger):
    """
    Modifier logger that handles outputting values into a TensorBoard log directory
    for viewing in TensorBoard.

    :param log_path: the path to create a SummaryWriter at. writer must be None
        to use if not supplied (and writer is None),
        will create a TensorBoard dir in cwd
    :param writer: the writer to log results to,
        if none is given creates a new one at the log_path
    :param name: name given to the logger, used for identification;
        defaults to tensorboard
    """

    def __init__(
        self,
        log_path: str = None,
        writer: SummaryWriter = None,
        name: str = "tensorboard",
    ):
        super().__init__(name)
        if tensorboard_import_error:
            raise tensorboard_import_error

        if writer and log_path:
            raise ValueError(
                (
                    "log_path given:{} and writer object passed in, "
                    "to create a writer at the log path set writer=None"
                ).format(log_path)
            )
        elif not writer and not log_path:
            log_path = os.path.join(".", "tensorboard")

        if log_path:
            create_dirs(log_path)

        self._writer = writer if writer is not None else SummaryWriter(log_path)

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        try:
            self._writer.add_hparams(params, {})
        except Exception:
            # fall back incase add_hparams isn't available, log as scalars
            for name, val in params.items():
                self.log_scalar(name, val)

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
        self._writer.add_scalar(tag, value, step, wall_time)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        self._writer.add_scalars(tag, values, step, wall_time)

    def log_histogram(
        self,
        tag: str,
        values: Union[Tensor, ndarray],
        bins: str = "tensorflow",
        max_bins: Union[int, None] = None,
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the histogram with
        :param values: values to log as a histogram
        :param bins: the type of bins to use for grouping the values,
            follows tensorboard terminology
        :param max_bins: maximum number of bins to use (default None)
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        self._writer.add_histogram(tag, values, step, bins, wall_time, max_bins)

    def log_histogram_raw(
        self,
        tag: str,
        min_val: Union[float, int],
        max_val: Union[float, int],
        num_vals: int,
        sum_vals: Union[float, int],
        sum_squares: Union[float, int],
        bucket_limits: Union[Tensor, ndarray],
        bucket_counts: Union[Tensor, ndarray],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the histogram with
        :param min_val: min value
        :param max_val: max value
        :param num_vals: number of values
        :param sum_vals: sum of all the values
        :param sum_squares: sum of the squares of all the values
        :param bucket_limits: upper value per bucket
        :param bucket_counts: number of values per bucket
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        self._writer.add_histogram_raw(
            tag,
            min_val,
            max_val,
            num_vals,
            sum_vals,
            sum_squares,
            bucket_limits,
            bucket_counts,
            step,
            wall_time,
        )
