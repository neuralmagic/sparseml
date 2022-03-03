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

import os
from typing import Callable

import pytest

from sparseml.keras.optim import (
    KerasModifierYAML,
    Modifier,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.keras.utils import keras
from sparseml.utils import KERAS_FRAMEWORK
from tests.sparseml.keras.optim.mock import mnist_model
from tests.sparseml.optim.test_modifier import (
    BaseModifierTest,
    BaseScheduledTest,
    BaseUpdateTest,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_KERAS_TESTS", False),
    reason="Skipping keras tests",
)
class ModifierTest(BaseModifierTest):
    # noinspection PyMethodOverriding
    def test_constructor(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_constructor(modifier_lambda, framework=KERAS_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_yaml(modifier_lambda, framework=KERAS_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml_key(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_yaml_key(modifier_lambda, framework=KERAS_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_repr(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_repr(modifier_lambda, framework=KERAS_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_props(modifier_lambda, framework=KERAS_FRAMEWORK)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_KERAS_TESTS", False),
    reason="Skipping keras tests",
)
class ScheduledModifierTest(ModifierTest, BaseScheduledTest):
    # noinspection PyMethodOverriding
    def test_props_start(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_props_start(modifier_lambda, framework=KERAS_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props_end(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_props_end(modifier_lambda, framework=KERAS_FRAMEWORK)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_KERAS_TESTS", False),
    reason="Skipping keras tests",
)
class ScheduledUpdateModifierTest(ScheduledModifierTest, BaseUpdateTest):
    # noinspection PyMethodOverriding
    def test_props_frequency(
        self,
        modifier_lambda: Callable[[], ScheduledUpdateModifier],
        model_lambda: Callable[[], keras.models.Model],
        steps_per_epoch: int,
    ):
        super().test_props_frequency(modifier_lambda, framework=KERAS_FRAMEWORK)


@KerasModifierYAML()
class ModifierImpl(Modifier):
    def __init__(self):
        super().__init__()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_KERAS_TESTS", False),
    reason="Skipping keras tests",
)
@pytest.mark.parametrize("modifier_lambda", [ModifierImpl], scope="function")
@pytest.mark.parametrize("model_lambda", [mnist_model], scope="function")
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestModifierImpl(ModifierTest):
    pass


@KerasModifierYAML()
class ScheduledModifierImpl(ScheduledModifier):
    def __init__(
        self,
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
    ):
        super().__init__()


@pytest.mark.parametrize("modifier_lambda", [ScheduledModifierImpl], scope="function")
@pytest.mark.parametrize("model_lambda", [mnist_model], scope="function")
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestScheduledModifierImpl(ScheduledModifierTest):
    pass


@KerasModifierYAML()
class ScheduledUpdateModifierImpl(ScheduledUpdateModifier):
    def __init__(
        self,
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
        update_frequency: float = -1,
    ):
        super().__init__()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_KERAS_TESTS", False),
    reason="Skipping keras tests",
)
@pytest.mark.parametrize(
    "modifier_lambda", [ScheduledUpdateModifierImpl], scope="function"
)
@pytest.mark.parametrize("model_lambda", [mnist_model], scope="function")
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestScheduledUpdateModifierImpl(ScheduledUpdateModifierTest):
    pass
