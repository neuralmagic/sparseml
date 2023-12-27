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


from typing import Literal, Union


__all__ = [
    "FrequencyManager",
    "PossibleLoggingMode",
    "PossibleFrequencyType",
    "LogStepType",
]

LogStepType = Union[int, float, None]
PossibleLoggingMode = Literal["on_change", "exact"]
PossibleFrequencyType = Literal["epoch", "step", "batch"]


class FrequencyManager:
    """
    Class for managing the frequency of logging and model updates

    :param log_frequency: The frequency to log at
    :param mode: The logging mode to use, either "on_change" or "exact",
        "on_change" will log when the model has been updated since the last log,
        "exact" will log at the given frequency regardless of model updates
    :param frequency_type: The frequency type to use, either "epoch", "step", or "batch"
        controls what the frequency manager is tracking, e.g. if the frequency type
        is "epoch", then the frequency manager will track the number of epochs that
        have passed since the last log, if the frequency type is "step", then the
        frequency manager will track the number of optimizer steps
    """

    def __init__(
        self,
        log_frequency: LogStepType = None,
        mode: PossibleLoggingMode = "exact",
        frequency_type: PossibleFrequencyType = "epoch",
    ):
        # sets self._logging_mode and self._check_model_update
        self._logging_mode = self._set_logging_mode(mode=mode)

        # sets self._frequency_type and self._valid_python_types
        self._frequency_type = self._set_frequency_type(frequency_type=frequency_type)

        self._validate_log_frequency(log_frequency=log_frequency)
        self._log_frequency = log_frequency

        self.last_log_step: LogStepType = None
        self.last_model_update_step: LogStepType = None

    def log_ready(
        self,
        current_log_step: LogStepType,
        check_model_update: bool = False,
    ):
        """
        Check if the frequency manager is ready to log
        Conditions for readiness:
            - log frequency is not None
            - current log step is None
            - current log step greater than or equal to the last log step
                plus the log frequency
            - if check_model_update is True, or self._check_model_update is True,
                then the last model update step must be greater than or equal
                to the last log step, and the current log step must be greater
                than or equal to the last model update step plus the log frequency

        :param current_log_step: The current log step
        :param check_model_update: If True, will check if the model has been updated
            since the last log step and if _log_frequency steps have passed since the
            last model update; Defaults to False.
        :return: True if the frequency manager is ready to log,
            False otherwise
        """
        # check_model_update is used to override self._check_model_update
        # e.g. if check_model_update is True, then the model update check
        # will be performed even if self._check_model_update is False

        check_model_update = check_model_update or self._check_model_update

        # format is used to avoid floating point errors
        # e.g. 0.1 + 0.2 != 0.3
        # format(0.1 + 0.2, ".4f") == format(0.3, ".4f")

        cadence_reached: bool = self._log_frequency is not None and (
            current_log_step is None
            or self.last_log_step is None
            or current_log_step
            >= float(format(self.last_log_step + self._log_frequency, ".4f"))
        )

        if not cadence_reached or not check_model_update:
            # early return if cadence not reached or,
            # model update check not requested
            return cadence_reached

        model_updated_since_last_log: bool = (
            self.last_model_update_step is None
            or self.last_log_step is None
            or current_log_step is None
            or (
                self.last_model_update_step >= self.last_log_step
                and current_log_step
                >= float(
                    format(self._log_frequency + self.last_model_update_step, ".4f")
                )
            )
        )

        return cadence_reached and model_updated_since_last_log

    def model_updated(self, step: LogStepType = None) -> None:
        """
        Sets the last model update to the given step

        :param step: The step to set the last model update to
        :post-cond: The last model update step is set to the given step
        """
        self._validate_log_step(log_step=step)
        self.last_model_update_step = step

    def log_written(self, step: LogStepType = None) -> None:
        """
        Sets the last log step to the given step

        :param step: The step to set the last log step to
        :post-cond: The last log step is set to the given step
        """
        self._validate_log_step(log_step=step)
        self.last_log_step = step

    @property
    def log_frequency(self) -> LogStepType:
        """
        :return: The log frequency
        """
        return self._log_frequency

    @log_frequency.setter
    def log_frequency(self, log_frequency: LogStepType) -> None:
        """
        Sets the log frequency to the given value

        :param log_frequency: The log frequency to set
        :post-cond: The log frequency is set to the given value
        """
        self._validate_log_frequency(log_frequency=log_frequency)
        self._log_frequency = log_frequency

    def _validate_log_frequency(self, log_frequency):
        # checks that log frequency is a positive number or None
        # raise TypeError if not a number or None
        # raises ValueError if not a positive number

        try:
            self._validate_log_step(log_step=log_frequency)
            if log_frequency == 0:
                raise ValueError()
            # except clauses update the error message
        except TypeError:
            raise TypeError(
                f"log frequency must be a number or None, given {type(log_frequency)}"
            )
        except ValueError:
            raise ValueError(
                f"log frequency must be greater than 0, given {log_frequency}"
            )

    def _validate_log_step(self, log_step):
        # checks that log step is a non negative number or None
        # raise TypeError if not a number or None
        # raises ValueError if negative number

        if not isinstance(log_step, self._valid_python_types) or isinstance(
            log_step, bool
        ):
            raise TypeError(
                f"log step must be a number or None, given {type(log_step)}"
            )

        if log_step is not None and log_step < 0:
            raise ValueError(
                f"log step must be greater than or equal to 0, given {log_step}"
            )

    def _set_logging_mode(self, mode: PossibleLoggingMode) -> PossibleLoggingMode:
        """
        Set the logging mode for the frequency manager.
        The logging mode determines how the frequency manager determines
        if it is ready to log

        :param mode: The logging mode to set
        :post-cond: The self._logging_mode is set to the given mode
        :post-cond: The self._check_model_update is set to True if the mode is
            "on_change"
        :raises ValueError: If the given mode is not one of "on_change" or "exact"
        :return: The logging mode that was set
        """
        mode = _basic_normalization(mode)
        if mode == "on_change":
            self._check_model_update = True
            self._logging_mode = "on_change"
        elif mode == "exact":
            self._check_model_update = False
            self._logging_mode = "exact"
        else:
            raise ValueError(
                f"Invalid logging mode {mode}, must be one of 'on_change', 'exact'"
            )
        return self._logging_mode

    def _set_frequency_type(
        self, frequency_type: PossibleFrequencyType
    ) -> PossibleFrequencyType:
        """
        Set the frequency type for the frequency manager.
        The frequency type determines what the frequency manager is tracking.
        For example, if the frequency type is "epoch", then the frequency manager
        will track the number of epochs that have passed since the last log.

        :param frequency_type: The frequency type to set
        :post-cond: The self._frequency_type is set to the given frequency type
        :post-cond: The self._valid_python_types is set to the valid python types
            for the given frequency type, e.g. (int, float, type(None)) for "epoch"
            and (int, type(None)) for "step" or "batch"
        :raises ValueError: If the given frequency type is not one of "epoch",
            "step", or "batch"
        :return: The frequency type that was set
        """
        frequency_type = _basic_normalization(frequency_type)
        if frequency_type == "epoch":
            self._frequency_type = "epoch"
            self._valid_python_types = (int, float, type(None))
        elif frequency_type == "step":
            self._frequency_type = "step"
            self._valid_python_types = (int, type(None))
        elif frequency_type == "batch":
            self._frequency_type = "batch"
            self._valid_python_types = (int, type(None))
        else:
            raise ValueError(
                f"Invalid frequency type {frequency_type}, must be one of "
                "'epoch', 'step', 'batch'"
            )
        return self._frequency_type


def _basic_normalization(value: str) -> str:
    """
    Basic normalization for string values.
    Removes leading and trailing whitespace and converts to lowercase.

    :param value: The value to normalize
    :return: The normalized value
    """
    return value.strip().lower()
