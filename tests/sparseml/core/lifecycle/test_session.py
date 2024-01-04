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
from types import SimpleNamespace

import pytest

import sparseml.core.session as session_manager
from sparseml.core import Framework
from sparseml.core.event import Event, EventType
from sparseml.core.lifecycle.event import CallbacksEventLifecycle
from sparseml.core.lifecycle.session import SparsificationLifecycle
from sparseml.core.modifier.base import ModifierInterface
from sparseml.core.state import State


def recipe_with_layer_prefix():
    layer_prefix = "model.decoder.layers"
    recipe = f"""
    metadata:
        target_model:
            layer_prefix: {layer_prefix}
            architecture: "opt"

    test_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                targets: __ALL_PRUNABLE__
                start: 0
                end: 5
    """
    return recipe, layer_prefix


def recipe_without_layer_prefix():
    recipe = """
    test_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                targets: __ALL_PRUNABLE__
                start: 0
                end: 5
    """
    return recipe, None


@pytest.fixture
def model():
    # identity model
    return lambda x: x


@pytest.mark.parametrize(
    "recipe, expected_layer_prefix",
    [
        recipe_without_layer_prefix(),
        recipe_with_layer_prefix(),
    ],
)
def test_session_initialize_propagates_layer_prefix_to_model(
    recipe, expected_layer_prefix, model
):
    session = session_manager.active_session()
    session.initialize(framework=Framework.general, model=model, recipe=recipe)
    print(f"{session.state.model.layer_prefix=}, {expected_layer_prefix=}")
    assert session.state.model.layer_prefix == expected_layer_prefix


class ModifierMock(ModifierInterface):
    initialized_ = False
    applied = False
    group = "test"
    unique_id = "test_0"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._hit_count = defaultdict(int)

    def initialized_structure(self) -> bool:
        self._hit_count["initialized_structure"] += 1
        pass

    def initialized(self) -> bool:
        self._hit_count["initialized"] += 1
        pass

    def finalized(self) -> bool:
        self._hit_count["finalized"] += 1
        pass

    def check_initialized(self):
        self._hit_count["check_initialized"] += 1
        pass

    def calculate_start(self) -> float:
        self._hit_count["calculate_start"] += 1
        pass

    def calculate_end(self) -> float:
        self._hit_count["calculate_end"] += 1
        pass

    def pre_initialize_structure(self, state: State, **kwargs):
        self._hit_count["pre_initialize_structure"] += 1
        return "pre_initialize_structure"

    def initialize(self, state: State, **kwargs):
        self._hit_count["initialize"] += 1
        return "initialize"

    def finalize(self, state: State, **kwargs):
        self._hit_count["finalize"] += 1
        return "finalize"

    def update_event(self, state: State, event: Event, **kwargs):
        self._hit_count["update_event"] += 1
        return "update_event"


class StateMock:
    def update(self, *args, **kwargs):
        return {"dummy": "dummy"}


def _empty_mock(*args, **kwargs):
    pass


class TestSparsificationLifecycle:
    @pytest.mark.parametrize(
        "lifecycle",
        [
            SparsificationLifecycle(state=State(framework=Framework.pytorch)),
        ],
    )
    @pytest.mark.parametrize("modifier_initialized", [True, False])
    @pytest.mark.parametrize("modifier_finalized", [True, False])
    def test_reset(
        self, lifecycle, modifier_initialized, modifier_finalized, monkeypatch
    ):
        monkeypatch.setattr(lifecycle, "modifiers", [ModifierMock()])
        monkeypatch.setattr(ModifierMock, "initialized_", modifier_initialized)
        monkeypatch.setattr(ModifierMock, "finalized", modifier_finalized)

        lifecycle.reset()

        empty_lifecycle = SparsificationLifecycle()
        assert lifecycle == empty_lifecycle

    @pytest.mark.parametrize(
        "lifecycle",
        [
            SparsificationLifecycle(),
        ],
    )
    @pytest.mark.parametrize(
        "method_name",
        [
            "pre_initialize_structure",
            "initialize",
        ],
    )
    def test_lifecycle_methods_call_modifier_methods(
        self, lifecycle, monkeypatch, method_name
    ):
        monkeypatch.setattr(lifecycle, "modifiers", [modifier_mock := ModifierMock()])
        monkeypatch.setattr(lifecycle, "_check_create_state", _empty_mock)
        monkeypatch.setattr(lifecycle, "_check_compile_recipe", _empty_mock)
        monkeypatch.setattr(lifecycle, "state", StateMock())

        method = getattr(lifecycle, method_name)
        results = method()

        assert modifier_mock._hit_count[method_name] == 1
        assert results == [method_name]

        if method_name == "pre_initialize_structure":
            assert lifecycle.initialized_structure
        else:
            assert lifecycle.initialized_

    @pytest.mark.parametrize(
        "initialized_, finalized",
        [
            (False, False),
            (False, True),
            (True, True),
        ],
    )
    def test_finalize_raises_value_error_if_not_initialized(
        self, initialized_, finalized, monkeypatch
    ):
        lifecycle = SparsificationLifecycle()
        lifecycle.initialized_ = initialized_

        monkeypatch.setattr(lifecycle, "finalized", finalized)

        with pytest.raises(ValueError, match="Cannot finalize"):
            lifecycle.finalize()

    def test_finalize_calls_modifier_finalize(self, monkeypatch):
        lifecycle = SparsificationLifecycle()
        lifecycle.initialized_ = True
        lifecycle.finalized = False

        monkeypatch.setattr(lifecycle, "modifiers", [modifier_mock := ModifierMock()])
        results = lifecycle.finalize()

        # assert lifecycle is finalized
        assert lifecycle.finalized

        assert modifier_mock._hit_count["finalize"] == 1
        assert results == ["finalize"]

    @pytest.mark.parametrize(
        "initialized_, finalized, event_type, kwargs",
        [
            (False, False, EventType.BATCH_START, {}),
            (False, True, EventType.BATCH_START, {}),
            (True, True, EventType.BATCH_START, {}),
            (True, False, EventType.PRE_INIT, {}),
            (True, False, EventType.INITIALIZE, {}),
            (True, False, EventType.FINALIZE, {}),
            (True, False, EventType.FINALIZE, {}),
            (True, False, EventType.LOSS_CALCULATED, {}),
            (True, False, EventType.LOSS_CALCULATED, {"loss": None}),
        ],
    )
    def test_event_raises_value_error(
        self, initialized_, finalized, monkeypatch, event_type, kwargs
    ):
        lifecycle = SparsificationLifecycle()
        lifecycle.initialized_ = initialized_

        monkeypatch.setattr(lifecycle, "finalized", finalized)

        with pytest.raises(ValueError):
            lifecycle.event(event_type=event_type, **kwargs)

    def test_event_sets_state_start_event(self, monkeypatch):

        lifecycle = SparsificationLifecycle(
            state=State(framework=Framework.pytorch),
            event_lifecycle=CallbacksEventLifecycle(
                type_first=EventType.BATCH_START, start=Event()
            ),
        )
        lifecycle.initialized_ = True
        lifecycle.finalized = False

        event_type = EventType.BATCH_START
        event = Event()

        def events_from_type_mock(*args, **kwargs):
            return [event]

        monkeypatch.setattr(lifecycle, "_check_setup_event_lifecycle", _empty_mock)
        monkeypatch.setattr(
            lifecycle.event_lifecycle, "events_from_type", events_from_type_mock
        )

        results = lifecycle.event(event_type=event_type)
        assert lifecycle.state.start_event == event
        assert lifecycle.state.last_event == event
        assert lifecycle.event_called
        assert results == []

    def test_event_calls_modifier_update_event(self, monkeypatch):
        lifecycle = SparsificationLifecycle(
            state=State(framework=Framework.pytorch),
            event_lifecycle=CallbacksEventLifecycle(
                type_first=EventType.BATCH_START, start=Event()
            ),
        )
        lifecycle.initialized_ = True
        lifecycle.finalized = False

        event_type = EventType.BATCH_START
        event = Event()

        def events_from_type_mock(*args, **kwargs):
            return [event]

        monkeypatch.setattr(lifecycle, "_check_setup_event_lifecycle", _empty_mock)
        monkeypatch.setattr(lifecycle, "modifiers", [modifier_mock := ModifierMock()])
        monkeypatch.setattr(
            lifecycle.event_lifecycle, "events_from_type", events_from_type_mock
        )

        results = lifecycle.event(event_type=event_type)
        assert modifier_mock._hit_count["update_event"] == 1
        assert results == ["update_event"]

    @pytest.mark.parametrize(
        "event_type",
        [
            EventType.BATCH_START,
            EventType.LOSS_CALCULATED,
            EventType.OPTIM_PRE_STEP,
            EventType.OPTIM_POST_STEP,
        ],
    )
    def test__check_setup_event_lifecycle(self, event_type, monkeypatch):
        lifecycle = SparsificationLifecycle()
        event = Event()

        class StateMock:
            model = 1
            start_event = 1
            sparsification_ready = 1
            start_event = event

        recipe_container_mock = SimpleNamespace(compiled_recipe=1)

        monkeypatch.setattr(lifecycle, "state", StateMock())
        monkeypatch.setattr(lifecycle, "recipe_container", recipe_container_mock)
        monkeypatch.setattr(lifecycle, "modifiers", [modifier_mock := ModifierMock()])

        lifecycle._check_setup_event_lifecycle(event_type=event_type)

        assert modifier_mock._hit_count["check_initialized"] == 1
        assert isinstance(lifecycle.event_lifecycle, CallbacksEventLifecycle)
        assert lifecycle.event_lifecycle.type_first == event_type
