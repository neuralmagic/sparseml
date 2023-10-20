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

from sparseml.core.event import Event, EventType
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
    def test__log_epoch(self, model_logging_mixin, state_mock):
        model_logging_mixin._log_epoch(state_mock.logger_manager, state_mock.epoch)
        assert state_mock.logger_manager.hit_count["log_string"] == 1
