"""
Contains code for loggers that help visualize the information from each modifier
"""

from abc import ABC, abstractmethod
from typing import Union, Dict
import time
from logging import Logger
from numpy import ndarray

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


__all__ = ["ModifierLogger", "PythonLogger", "TensorboardLogger"]


class ModifierLogger(ABC):
    """
    Base class that all modifier loggers must implement
    """

    def __init__(self, name: str):
        """
        :param name: name given to the logger, used for identification
        """
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
        :param params: Each key-value pair in the dictionary is the name of the hyper parameter and
                       it's corresponding value.
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
        :param bins: the type of bins to use for grouping the values, follows tensorboard terminology
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


class PythonLogger(ModifierLogger):
    """
    Modifier logger that handles printing values into a python logger instance
    """

    def __init__(self, logger: Logger = None, name: str = "python"):
        """
        :param logger: a logger instance to log to, if None then will create it's own
        :param name: name given to the logger, used for identification; defaults to python
        """
        super().__init__(name)
        self._logger = (
            logger if logger is not None else Logger("NeuralMagicModifiersLogger")
        )

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the hyper parameter and
                       it's corresponding value.
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
        :param wall_time: global wall time for when the value was taken, defaults to time.time()
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
        :param wall_time: global wall time for when the values were taken, defaults to time.time()
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
        :param bins: the type of bins to use for grouping the values, follows tensorboard terminology
        :param max_bins: maximum number of bins to use (default None)
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken, defaults to time.time()
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
        :param wall_time: global wall time for when the values were taken, defaults to time.time()
        """
        if wall_time is None:
            wall_time = time.time()

        msg = "{}-HISTOGRAM {} [{} - {}]: cannot log".format(
            self.name, tag, step, wall_time
        )
        self._logger.info(msg)


class TensorboardLogger(ModifierLogger):
    """
    Modifier logger that handles outputting values into a tensorboard log directory for viewing in tensorboard
    """

    def __init__(self, writer: SummaryWriter = None, name: str = "tensorboard"):
        """
        :param writer: the writer to log results to, if none is given creates a new one at the current working dir
        :param name: name given to the logger, used for identification; defaults to tensorboard
        """
        super().__init__(name)
        self._writer = writer if writer is not None else SummaryWriter()

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the hyper parameter and
                       it's corresponding value.
        """
        self._writer.add_hparams(params, {})

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
        :param wall_time: global wall time for when the value was taken, defaults to time.time()
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
        :param wall_time: global wall time for when the values were taken, defaults to time.time()
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
        :param bins: the type of bins to use for grouping the values, follows tensorboard terminology
        :param max_bins: maximum number of bins to use (default None)
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken, defaults to time.time()
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
        :param wall_time: global wall time for when the values were taken, defaults to time.time()
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
