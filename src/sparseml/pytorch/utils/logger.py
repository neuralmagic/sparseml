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
from abc import ABC
from logging import Logger
from typing import Callable, Dict, List, Optional, Union


try:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ModuleNotFoundError, ImportError):
        from tensorboardX import SummaryWriter
    tensorboard_import_error = None
except Exception as tensorboard_err:
    SummaryWriter = object
    tensorboard_import_error = tensorboard_err


try:
    import wandb

    wandb_err = None
except Exception as err:
    wandb = None
    wandb_err = err

from sparseml.utils import create_dirs


__all__ = [
    "BaseLogger",
    "LambdaLogger",
    "PythonLogger",
    "TensorBoardLogger",
    "WANDBLogger",
    "SparsificationGroupLogger",
]


class BaseLogger(ABC):
    """
    Base class that all modifier loggers must implement.

    :param name: name given to the logger, used for identification
    :param enabled: True to log, False otherwise
    """

    def __init__(self, name: str, enabled: bool = True):
        self._name = name
        self._enabled = enabled

    @property
    def name(self) -> str:
        """
        :return: name given to the logger, used for identification
        """
        return self._name

    @property
    def enabled(self) -> bool:
        """
        :return: True to log, False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to log, False otherwise
        """
        self._enabled = value

    def log_hyperparams(self, params: Dict[str, float]) -> bool:
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        :return: True if logged, False otherwise.
        """
        return False

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        wall_time: Optional[float] = None,
    ) -> bool:
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken
        :return: True if logged, False otherwise.
        """
        return False

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
        :return: True if logged, False otherwise.
        """
        return False


class LambdaLogger(BaseLogger):
    """
    Logger that handles calling back to a lambda function with any logs.

    :param lambda_func: the lambda function to call back into with any logs.
        The expected call sequence is (tag, value, values, step, wall_time) -> bool
        The return type is True if logged and False otherwise.
    :param name: name given to the logger, used for identification;
        defaults to lambda
    :param enabled: True to log, False otherwise
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
            bool,
        ],
        name: str = "lambda",
        enabled: bool = True,
    ):
        super().__init__(name, enabled)
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
        bool,
    ]:
        """
        :return: the lambda function to call back into with any logs.
            The expected call sequence is (tag, value, values, step, wall_time)
        """
        return self._lambda_func

    def log_hyperparams(self, params: Dict) -> bool:
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        :return: True if logged, False otherwise.
        """
        if not self.enabled:
            return False

        return self._lambda_func(None, None, params, None, None)

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
        :return: True if logged, False otherwise.
        """
        if not self.enabled:
            return False

        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(tag, value, None, step, wall_time)

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
        :return: True if logged, False otherwise.
        """
        if not self.enabled:
            return False

        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(tag, None, values, step, wall_time)


class PythonLogger(LambdaLogger):
    """
    Modifier logger that handles printing values into a python logger instance.

    :param logger: a logger instance to log to, if None then will create it's own
    :param log_level: level to log any incoming data at on the logging.Logger instance
    :param name: name given to the logger, used for identification;
        defaults to python
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        logger: Logger = None,
        log_level: int = logging.INFO,
        name: str = "python",
        enabled: bool = True,
    ):
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._log_level = log_level
        super().__init__(lambda_func=self._log_lambda, name=name, enabled=enabled)

    def __getattr__(self, item):
        return getattr(self._logger, item)

    @property
    def logger(self) -> Logger:
        """
        :return: a logger instance to log to, if None then will create it's own
        """
        return self._logger

    def _log_lambda(
        self,
        tag: Optional[str],
        value: Optional[float],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
    ) -> bool:
        if not values:
            values = {}

        if value:
            values["__value__"] = value

        self._logger.log(
            self._log_level,
            "%s %s [%s - %s]: %s",
            self.name,
            tag,
            step,
            wall_time,
            values,
        )

        return True


class TensorBoardLogger(LambdaLogger):
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
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        log_path: str = None,
        writer: SummaryWriter = None,
        name: str = "tensorboard",
        enabled: bool = True,
    ):
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
        super().__init__(lambda_func=self._log_lambda, name=name, enabled=enabled)

    @property
    def writer(self) -> SummaryWriter:
        """
        :return: the writer to log results to,
            if none is given creates a new one at the log_path
        """
        return self._writer

    def _log_lambda(
        self,
        tag: Optional[str],
        value: Optional[float],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
    ) -> bool:
        if value is not None:
            self._writer.add_scalar(tag, value, step, wall_time)

        if values and tag:
            self._writer.add_scalars(tag, values, step, wall_time)
        elif values:
            for name, val in values.items():
                # hyperparameters logging case
                self._writer.add_scalar(name, val, step, wall_time)

        return True


class WANDBLogger(LambdaLogger):
    """
    Modifier logger that handles outputting values to Weights and Biases.

    :param init_kwargs: the args to call into wandb.init with;
        ex: wandb.init(**init_kwargs). If not supplied, then init will not be called
    :param name: name given to the logger, used for identification;
        defaults to wandb
    :param enabled: True to log, False otherwise
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
        enabled: bool = True,
    ):
        super().__init__(lambda_func=self._log_lambda, name=name, enabled=enabled)

        if wandb_err:
            raise wandb_err

        if init_kwargs:
            wandb.init(**init_kwargs)

    def _log_lambda(
        self,
        tag: Optional[str],
        value: Optional[float],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
    ) -> bool:
        params = {}

        if value is not None:
            params[tag] = value

        if values:
            if tag:
                values = {f"{tag}/{key}": val for key, val in values.items()}
            params.update(values)

        wandb.log(params, step=step)

        return True


class SparsificationGroupLogger(BaseLogger):
    """
    Modifier logger that handles outputting values to other supported systems.
    Supported ones include:
      - Python logging
      - Tensorboard
      - Weights and Biases
      - Lambda callback

    All are optional and can be bulk disabled and enabled by this root.

    :param lambda_func: an optional lambda function to call back into with any logs.
        The expected call sequence is (tag, value, values, step, wall_time) -> bool
        The return type is True if logged and False otherwise.
    :param python: an optional argument for logging to a python logger.
        May be a logging.Logger instance to log to, True to create a logger instance,
        or non truthy to not log anything (False, None)
    :param python_log_level: if python,
        the level to log any incoming data at on the logging.Logger instance
    :param tensorboard: an optional argument for logging to a tensorboard writer.
        May be a SummaryWriter instance to log to, a string representing the directory
        to create a new SummaryWriter to log to, True to create a new SummaryWriter,
        or non truthy to not log anything (False, None)
    :param wandb_: an optional argument for logging to wandb.
        May be a dictionary to pass to the init call for wandb,
        True to log to wandb (will not call init),
        or non truthy to not log anything (False, None)
    :param name: name given to the logger, used for identification;
        defaults to sparsification
    :param enabled: True to log, False otherwise
    """

    def __init__(
        self,
        lambda_func: Optional[
            Callable[
                [
                    Optional[str],
                    Optional[float],
                    Optional[Dict[str, float]],
                    Optional[int],
                    Optional[float],
                ],
                bool,
            ]
        ] = None,
        python: Optional[Union[bool, Logger]] = None,
        python_log_level: int = logging.INFO,
        tensorboard: Optional[Union[bool, str, SummaryWriter]] = None,
        wandb_: Optional[Union[bool, Dict]] = None,
        name: str = "sparsification",
        enabled: bool = True,
    ):
        super().__init__(name, enabled)
        self._loggers: List[BaseLogger] = []

        if lambda_func:
            self._loggers.append(
                LambdaLogger(lambda_func=lambda_func, name=name, enabled=enabled)
            )

        if python:
            self._loggers.append(
                PythonLogger(
                    logger=python if isinstance(python, Logger) else None,
                    log_level=python_log_level,
                    name=name,
                    enabled=enabled,
                )
            )

        if tensorboard:
            self._loggers.append(
                TensorBoardLogger(
                    log_path=tensorboard if isinstance(tensorboard, str) else None,
                    writer=(
                        tensorboard if isinstance(tensorboard, SummaryWriter) else None
                    ),
                    name=name,
                    enabled=enabled,
                )
            )

        if wandb_ and WANDBLogger.available():
            self._loggers.append(
                WANDBLogger(
                    init_kwargs=wandb_ if isinstance(wandb_, Dict) else None,
                    name=name,
                    enabled=enabled,
                )
            )

    @BaseLogger.enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to log, False otherwise
        """
        self._enabled = value

        for logger in self._loggers:
            logger.enabled = value

    @property
    def loggers(self) -> List[BaseLogger]:
        """
        :return: the created logger sub instances for this logger
        """
        return self._loggers

    def log_hyperparams(self, params: Dict):
        """
        :param params: Each key-value pair in the dictionary is the name of the
            hyper parameter and it's corresponding value.
        """
        for logger in self._loggers:
            logger.log_hyperparams(params)

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
        for logger in self._loggers:
            logger.log_scalar(tag, value, step, wall_time)

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
        for logger in self._loggers:
            logger.log_scalars(tag, values, step, wall_time)
