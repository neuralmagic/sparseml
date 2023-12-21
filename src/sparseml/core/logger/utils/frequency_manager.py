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


from typing import Union


__all__ = ["FrequencyManager"]

LogStepType = Union[int, float, None]


class FrequencyManager:
    """
    Class for managing the frequency of logging and model updates

    :param log_frequency: The frequency to log at
    """

    def __init__(self, log_frequency: LogStepType = None):
        self._validate_log_frequency(log_frequency=log_frequency)
        self.log_frequency = log_frequency
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
            - if check_model_update is True, then the last model update step
                must be greater than or equal to the last log step, and the current
                log step must be greater than or equal to the last model update step
                plus the log frequency

        :param current_log_step: The current log step
        :param check_model_update: If True, will check if the model has been updated
            since the last log step and if _log_frequency steps have passed since the
            last model update; Defaults to False.
        :return: True if the frequency manager is ready to log,
            False otherwise
        """
        # format is used to avoid floating point errors
        # e.g. 0.1 + 0.2 != 0.3
        # format(0.1 + 0.2, ".4f") == format(0.3, ".4f")

        cadence_reached: bool = self.log_frequency is not None and (
            current_log_step is None
            or self.last_log_step is None
            or current_log_step
            >= float(format(self.last_log_step + self.log_frequency, ".4f"))
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
                    format(self.log_frequency + self.last_model_update_step, ".4f")
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

        if not isinstance(log_step, (int, float, type(None))) or isinstance(
            log_step, bool
        ):
            raise TypeError(
                f"log step must be a number or None, given {type(log_step)}"
            )

        if log_step is not None and log_step < 0:
            raise ValueError(
                f"log step must be greater than or equal to 0, given {log_step}"
            )
