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

from collections import defaultdict

import pytest

from sparseml.core.event import EventType
from sparseml.core.helpers import (
    _log_current_step,
    _log_model_loggable_items,
    log_model_info,
)
from sparseml.core.logger.logger import LoggerManager


class ModelMock:
    def loggable_items(self):
        for value in [("a", 1), ("b", 2), ("c", 3)]:
            yield value


class LoggerManagerMock:
    def __init__(self):
        self.hit_count = defaultdict(int)

    def epoch_to_step(self, epoch, steps_per_epoch):
        self.hit_count["epoch_to_step"] += 1
        return epoch * steps_per_epoch

    def log_string(self, tag, string, step):
        self.hit_count["log_string"] += 1

    def log_scalars(self, tag, values, step):
        self.hit_count["log_scalars"] += 1

    def log_scalar(self, tag, value, step):
        self.hit_count["log_scalar"] += 1

    def log_ready(self, *args, **kwargs):
        pass

    @property
    def __class__(self):
        return LoggerManager


class StateMock:
    def __init__(self, steps_per_epoch=10, epoch=1):
        self.model = ModelMock()
        self.loggers = LoggerManagerMock()
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch


class EventMock:
    def __init__(self, type_=EventType.BATCH_END, current_index=1, steps_per_epoch=10):
        self.type_ = type_
        self.current_index = current_index
        self.steps_per_epoch = steps_per_epoch


@pytest.fixture
def state_mock():
    yield StateMock()


def test__log_epoch_invokes_log_scalar():
    logger_manager = LoggerManagerMock()
    _log_current_step(
        logger_manager=logger_manager,
        current_log_step=1,
    )
    # log epoch should invoke log_string
    assert logger_manager.hit_count["log_scalar"] == 1


def test_log_model_info_logs_epoch_and_loggable_items():
    state = StateMock()
    epoch = 3
    log_model_info(state, current_log_step=epoch)

    # loggable items will invoke log_scalar for each
    # int/float value + 1 for the epoch
    assert state.loggers.hit_count["log_scalar"] == epoch + 1


@pytest.mark.parametrize(
    "loggable_items", [ModelMock().loggable_items(), [("a", {}), ("b", 2), ("c", {})]]
)
def test__log_model_loggable_items_routes_appropriately(loggable_items, monkeypatch):
    logger_manager = LoggerManagerMock()
    loggable_items = list(loggable_items)

    scalar_count = scalars_count = string_count = 0
    for _, value in loggable_items:
        if isinstance(value, (int, float)):
            scalar_count += 1
        elif isinstance(value, dict):
            scalars_count += 1
        else:
            string_count += 1

    _log_model_loggable_items(
        logger_manager=logger_manager, loggable_items=loggable_items, epoch=1
    )

    # loggable items will invoke log_scalar for each
    # int/float value
    assert logger_manager.hit_count["log_scalar"] == scalar_count

    # loggable items will invoke log_scalars for each
    # dict value
    assert logger_manager.hit_count["log_scalars"] == scalars_count

    # All other value types will invoke log_string
    assert logger_manager.hit_count["log_string"] == string_count
