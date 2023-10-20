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
from sparseml.core.logger import LoggerManager
from sparseml.core.modifier.mixins import ModelLoggingMixin


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
def model_logging_mixin():
    yield ModelLoggingMixin()


@pytest.fixture
def state_mock():
    yield StateMock()


class TestModelLoggingMixin:
    def test__log_epoch(self, model_logging_mixin):
        logger_manager = LoggerManagerMock()
        model_logging_mixin._log_epoch(
            logger_manager=logger_manager,
            epoch=1,
        )
        assert logger_manager.hit_count["log_string"] == 1

    @pytest.mark.parametrize(
        "type_, current_index, expected",
        [
            (EventType.BATCH_END, 1, True),
            (EventType.BATCH_END, 10, True),
            (EventType.BATCH_END, 1.3, False),
            (EventType.BATCH_START, 1, False),
            (EventType.OPTIM_POST_STEP, 1, False),
            (EventType.OPTIM_PRE_STEP, 1, False),
        ],
    )
    def test__should_log_model_info(
        self, model_logging_mixin, type_, current_index, expected
    ):
        state = StateMock()
        event = EventMock(type_=type_, current_index=current_index)
        assert model_logging_mixin._should_log_model_info(state, event) == expected

    def test_log_model_info(self, model_logging_mixin, monkeypatch):
        state = StateMock()
        event = EventMock()
        monkeypatch.setattr(
            model_logging_mixin, "_should_log_model_info", lambda *args, **kwargs: True
        )
        monkeypatch.setattr(
            model_logging_mixin, "_log_epoch", lambda *args, **kwargs: None
        )

        model_logging_mixin.log_model_info(state, event)
        assert state.loggers.hit_count["log_string"] == 3
