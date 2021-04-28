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
from typing import Callable, Dict, Optional, Union


try:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ModuleNotFoundError, ImportError):
        from tensorboardX import SummaryWriter
    tensorboard_import_error = None
except Exception as tensorboard_err:
    SummaryWriter = None
    tensorboard_import_error = tensorboard_err


try:
    import wandb

    wandb_error = None
except Exception as wandb_err:
    wandb = None
    wandb_err = wandb_err

from sparseml.utils import create_dirs


__all__ = [
    "PyTorchLogger",
    "LambdaLogger",
    "PythonLogger",
    "TensorBoardLogger",
    "WAndBLogger",
]


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
    def log_hyperparams(self, params: Dict[str, float]):
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
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
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
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken
        """
        raise NotImplementedError()


class LambdaLogger(PyTorchLogger):
    """
    Logger that handles calling back to a lambda function with any logs.

    :param lambda_func: the lambda function to call back into with any logs.
        The expected call sequence is (tag, value, values, step, wall_time)
    :param name: name given to the logger, used for identification;
        defaults to lambda
    """

    def __init__(
        self,
        lambda_func: Callable[
            [
                Optional[str],
                Optional[float],
                Optional[Dict[str, float]],
                Optional[int],
                Optional[float],
            ],
            None,
        ],
        name: str = "lambda",
    ):
        super().__init__(name)
        self._lambda_func = lambda_func
        assert lambda_func, "lambda_func must be set to a callable function"

    @property
    def lambda_func(
        self,
    ) -> Callable[
        [
            Optional[str],
            Optional[float],
            Optional[Dict[str, float]],
            Optional[int],
            Optional[float],
        ],
        None,
    ]:
        """
        :return: the lambda function to call back into with any logs.
            The expected call sequence is (tag, value, values, step, wall_time)
        """
        return self._lambda_func

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        self._lambda_func(None, None, params, None, None)

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
        if not wall_time:
            wall_time = time.time()
        self._lambda_func(tag, value, None, step, wall_time)

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
        if not wall_time:
            wall_time = time.time()
        self._lambda_func(tag, None, values, step, wall_time)


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

    @property
    def logger(self) -> Logger:
        """
        :return: a logger instance to log to, if None then will create it's own
        """
        return self._logger

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
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
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
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
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

    @property
    def writer(self) -> SummaryWriter:
        """
        :return: the writer to log results to,
            if none is given creates a new one at the log_path
        """
        return self._writer

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
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
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
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        self._writer.add_scalars(tag, values, step, wall_time)


class WAndBLogger(PyTorchLogger):
    """
    Modifier logger that handles outputting values to Weights and Biases.

    :param init_kwargs: the args to call into wandb.init with;
        ex: wandb.init(**init_kwargs). If not supplied, then init will not be called
    :param name: name given to the logger, used for identification;
        defaults to wandb
    """

    @staticmethod
    def available() -> bool:
        """
        :return: True if wandb is available and installed, False, otherwise
        """
        return not wandb_err

    def __init__(
        self,
        init_kwargs: Optional[Dict] = None,
        name: str = "wandb",
    ):
        super().__init__(name)

        if wandb_err:
            raise wandb_err

        if init_kwargs:
            wandb.init(**init_kwargs)

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        wandb.log(params)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken,
            defaults to time.time()
        """
        wandb.log({tag: value}, step=step)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        """
        wandb.log({tag: values}, step=step)
