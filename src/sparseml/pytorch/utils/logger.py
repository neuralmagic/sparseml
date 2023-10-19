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


# This file has been moved to src/sparseml/core/logger.py
# and is kept here for backwards compatibility.
# It will be removed in a future release.

from sparseml.core.logger import (
    LOGGING_LEVELS,
    BaseLogger,
    LambdaLogger,
    LoggerManager,
    PythonLogger,
    SparsificationGroupLogger,
    TensorBoardLogger,
    WANDBLogger,
)


__all__ = [
    "BaseLogger",
    "LambdaLogger",
    "PythonLogger",
    "TensorBoardLogger",
    "WANDBLogger",
    "SparsificationGroupLogger",
    "LoggerManager",
    "LOGGING_LEVELS",
]
