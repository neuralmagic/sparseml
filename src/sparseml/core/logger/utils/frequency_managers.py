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

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Union


__all__ = [
    "FrequencyManagerContract",
    "EpochFrequencyManager",
    "BatchFrequencyManager",
    "OptimizerStepFrequencyManager",
    "FrequencyManagerFactory",
]


class FrequencyManagerContract(ABC):
    """
    Contract for frequency managers
    """

    def __init__(
        self,
        log_frequency: Union[float, int, None] = None,
        frequency_type: str = "epoch",
    ):
        self.validate_log_frequency(log_frequency=log_frequency)
        self._log_frequency = log_frequency
        self._frequency_type = frequency_type

    @abstractmethod
    def validate_log_frequency(self, log_frequency):
        """
        Validates the log frequency, raising a ValueError if invalid
        :param log_frequency: The log frequency to validate
        """
        raise NotImplementedError

    @abstractmethod
    def log_ready(self, current_log_step, last_log_step) -> bool:
        """
        :param current_log_step: The current log step
        :param last_log_step: The last step we logged at
        :return: True if the frequency manager is ready to log, False otherwise
        """
        raise NotImplementedError

    @property
    def frequency_type(self) -> str:
        return self._frequency_type


class EpochFrequencyManager(FrequencyManagerContract):
    """
    Frequency manager that handles logging based on epochs.
    Can accept a float or int for the frequency, where a float is a
    fraction of an epoch

    :param log_frequency: The frequency to log at, either a float or int or None
    """

    def log_ready(
        self,
        current_log_step: Union[int, float, None],
        last_log_step: Union[int, float, None],
    ):
        """
        :param current_log_step: The current log step
        :param last_log_step: The last step we logged at
        :return: True if the frequency manager is ready to log,
            False otherwise
        """
        return self._log_frequency is not None and (
            current_log_step is None
            or last_log_step is None
            or current_log_step >= last_log_step + self._log_frequency
        )

    def validate_log_frequency(self, log_frequency: Union[float, int, None]):
        """
        Validates the log frequency is a float or int or None, raising
        a ValueError if invalid

        :param log_frequency: _description_
        :raises ValueError: _description_
        """
        if not isinstance(log_frequency, (float, int, type(None))):
            raise ValueError(
                f"frequency {log_frequency} must be a float or int or None"
                f" But found {type(log_frequency)} instead"
            )

    @property
    def log_frequency(self) -> Union[float, int, None]:
        return self._log_frequency

    @log_frequency.setter
    def log_frequency(self, value: Union[float, int, None]):
        self.validate_log_frequency(log_frequency=value)
        self._log_frequency = value


class IntegerFrequencyManager_(EpochFrequencyManager):
    """
    Frequency manager that handles logging based on integer steps only.
    """

    def validate_log_frequency(self, log_frequency: Optional[int]):
        """
        Validates the log frequency is an int or None, raising a ValueError if invalid

        :param log_frequency: the log frequency to validate
        :raises ValueError: if the log frequency is neither an int or None
        """
        if not isinstance(log_frequency, (int, type(None))):
            raise ValueError(f"frequency {log_frequency} must be an int or None")
        super().validate_log_frequency(log_frequency)


class BatchFrequencyManager(IntegerFrequencyManager_):
    """
    Frequency manager that handles logging based on batches.
    Accepts an int for the frequency, where the frequency is the number of batches
    between logs, or None for to log on every batch. If not None, the frequency
    must be >= 16
    """

    def validate_log_frequency(self, log_frequency: Optional[int]):
        """
        Validates the log frequency is an int or None, raising a ValueError if invalid
        Also validates that the frequency is >= 100 (to avoid over-logging) if not None

        :param log_frequency: the log frequency to validate
        :raises ValueError: if the log frequency is neither an int or None
        :raises ValueError: if the log frequency is < 10
        """
        super().validate_log_frequency(log_frequency)
        if log_frequency is not None and log_frequency < 10:
            raise ValueError(f"frequency {log_frequency} must be >= 10")


class OptimizerStepFrequencyManager(IntegerFrequencyManager_):
    """
    Frequency manager that handles logging based on optimizer steps.
    Accepts an int for the frequency, where the frequency is the number of optimizer
    steps between logs, or None for to log on every optimizer step. If not None,
    the frequency must be >=50
    """

    def validate_log_frequency(self, log_frequency: Optional[int]):
        """
        Validates the log frequency is an int or None, raising a ValueError if invalid
        Also validates that the frequency is >= 4 (to avoid over-logging) if not None

        :param log_frequency: the log frequency to validate
        :raises ValueError: if the log frequency is neither an int or None
        :raises ValueError: if the log frequency is < 4
        """
        super().validate_log_frequency(log_frequency)
        if log_frequency is not None and log_frequency < 4:
            raise ValueError(f"frequency {log_frequency} must be >= 4")


_CONSTRUCTORS = defaultdict(
    lambda: None,
    {
        "epoch": EpochFrequencyManager,
        "batch": BatchFrequencyManager,
        "optimizer_step": OptimizerStepFrequencyManager,
    },
)


class FrequencyManagerFactory:
    @staticmethod
    def from_frequency_type(
        frequency_type: str, log_frequency: Union[float, int, None] = None
    ) -> FrequencyManagerContract:
        """
        Factory method for creating a frequency manager based on the frequency type

        :param frequency_type: The frequency type to create, is case insensitive
        :param log_frequency: The frequency to log at, either a float or int or None;
            Default is None. If None, will log at every step
        :return: The frequency manager
        """

        normalized_frequency_type = frequency_type.lower().replace("-", "_").strip()
        if frequency_manager := _CONSTRUCTORS[normalized_frequency_type]:
            return frequency_manager(
                log_frequency=log_frequency, frequency_type=normalized_frequency_type
            )

        raise ValueError(
            f"Invalid frequency type {frequency_type}, must be one "
            f"of {list(_CONSTRUCTORS.keys())}"
        )
