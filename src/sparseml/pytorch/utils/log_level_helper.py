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


__all__ = [
    "convert_to_logging_level",
]

import sys


_LOG_LEVELS = [
    "OFF",
    "DEFAULT",
    "ON_LR_CHANGE",
    "ON_EPOCH_CHANGE",
    "ON_LR_OR_EPOCH_CHANGE",
]


def convert_to_logging_level(value: any) -> str:
    """
    :param value: the value to be converted to logging level,
        supports logical values as strings ie True, t, false, 0
        Any integer or floating point value < 0 or >= # of LOG LEVELS will default to
        'OFF'
    :return: a string representing the LOG LEVEL of the value,
        if it can't be determined, falls back on returning 'OFF'
    """
    if isinstance(value, str) and value.upper() in _LOG_LEVELS:
        return value.upper()
    if isinstance(value, float):
        value = int(max(min(value, sys.maxsize), -sys.maxsize))  # Could overflow occur?
    if isinstance(value, int):
        return _LOG_LEVELS[value] if 0 <= value < len(_LOG_LEVELS) else _LOG_LEVELS[0]

    # value is not a float or int or a defined LOG LEVEL find the truthy value
    result = (
        bool(value)
        if not isinstance(value, str)
        else bool(value) and (value.lower().strip() not in ["f", "0", "false"])
    )
    return _LOG_LEVELS[int(result)]
