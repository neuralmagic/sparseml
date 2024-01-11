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

import logging
import os
import time
from abc import ABC

import pytest

from sparseml.pytorch.utils import (
    LambdaLogger,
    LoggerManager,
    PythonLogger,
    SparsificationGroupLogger,
    TensorBoardLogger,
    WANDBLogger,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "logger",
    [
        PythonLogger(),
        TensorBoardLogger(),
        LambdaLogger(
            lambda_func=lambda tag, value, values, step, wall_time, level: logging.info(
                f"{tag}, {value}, {values}, {step}, {wall_time}, {level}"
            )
            or True
        ),
        *([WANDBLogger()] if WANDBLogger.available() else []),
        SparsificationGroupLogger(
            lambda_func=lambda tag, value, values, step, wall_time, level: logging.info(
                f"{tag}, {value}, {values}, {step}, {wall_time}, {level}"
            )
            or True,
            python=True,
            tensorboard=True,
            wandb_=True,
        ),
        LoggerManager(),
        LoggerManager(
            [
                TensorBoardLogger(),
                WANDBLogger() if WANDBLogger.available() else PythonLogger(),
            ]
        ),
    ],
)
class TestModifierLogger(ABC):
    def test_name(self, logger):
        assert logger.name is not None

    def test_log_hyperparams(self, logger):
        logger.log_hyperparams({"param1": 0.0, "param2": 1.0})
        logger.log_hyperparams({"param1": 0.0, "param2": 1.0}, level=10)

    def test_log_scalar(self, logger):
        logger.log_scalar("test-scalar-tag", 0.1)
        logger.log_scalar("test-scalar-tag", 0.1, 1)
        logger.log_scalar("test-scalar-tag", 0.1, 2, time.time() - 1)
        logger.log_scalar("test-scalar-tag", 0.1, 2, time.time() - 1, level=10)

    def test_log_scalars(self, logger):
        logger.log_scalars("test-scalars-tag", {"scalar1": 0.0, "scalar2": 1.0})
        logger.log_scalars("test-scalars-tag", {"scalar1": 0.0, "scalar2": 1.0}, 1)
        logger.log_scalars(
            "test-scalars-tag", {"scalar1": 0.0, "scalar2": 1.0}, 2, time.time() - 1
        )
        logger.log_scalars(
            "test-scalars-tag",
            {"scalar1": 0.0, "scalar2": 1.0},
            2,
            time.time() - 1,
            level=10,
        )
