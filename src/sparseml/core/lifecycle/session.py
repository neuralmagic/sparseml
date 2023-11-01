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

from dataclasses import dataclass, field
from typing import Any, List, Optional

from sparseml.core.event import EventType
from sparseml.core.framework import Framework
from sparseml.core.lifecycle.event import CallbacksEventLifecycle, EventLifecycle
from sparseml.core.modifier import StageModifiers
from sparseml.core.recipe import RecipeContainer
from sparseml.core.state import State


__all__ = [
    "SparsificationLifecycle",
]


@dataclass
class SparsificationLifecycle:
    state: Optional[State] = None
    recipe_container: RecipeContainer = field(default_factory=RecipeContainer)
    modifiers: List[StageModifiers] = field(default_factory=list)
    event_lifecycle: Optional[EventLifecycle] = None

    initialized_structure: bool = False
    initialized_: bool = False
    finalized: bool = False
    event_called: bool = False

    def reset(self):
        for mod in self.modifiers:
            if not mod.initialized or mod.finalized:
                continue

            try:
                mod.finalize(self.state)
            except Exception:
                pass

        if self.state and self.state.data:
            # reset data if it exists
            self.state.data.reset()

        self.state = None
        self.recipe_container = RecipeContainer()
        self.modifiers = []
        self.event_lifecycle = None

        self.initialized_structure = False
        self.initialized_ = False
        self.finalized = False
        self.event_called = False

    def pre_initialize_structure(
        self, framework: Framework = None, **kwargs
    ) -> List[Any]:
        self._check_create_state(framework=framework)
        extras = self.state.update(**kwargs)
        extras = self.recipe_container.update(**extras)

        self._check_compile_recipe()
        mod_data = []
        for mod in self.modifiers:
            data = mod.pre_initialize_structure(state=self.state, **extras)
            if data is not None:
                mod_data.append(data)

        self.initialized_structure = True

        return mod_data

    def initialize(self, framework: Framework = None, **kwargs) -> List[Any]:
        self._check_create_state(framework=framework)
        extras = self.state.update(**kwargs)
        extras = self.recipe_container.update(**extras)

        self._check_compile_recipe()
        self._set_model_layer_prefix()
        mod_data = []
        for mod in self.modifiers:
            data = mod.initialize(state=self.state, **extras)
            if data is not None:
                mod_data.append(data)

        self.initialized_ = True

        return mod_data

    def finalize(self, **kwargs) -> List[Any]:
        if not self.initialized_:
            raise ValueError("Cannot finalize before initializing")

        if self.finalized:
            raise ValueError("Cannot finalize more than once")

        mod_data = []
        for mod in self.modifiers:
            data = mod.finalize(state=self.state, **kwargs)
            if data is not None:
                mod_data.append(data)

        self.finalized = True

        return mod_data

    def event(self, event_type: EventType, **kwargs) -> List[Any]:
        if not self.initialized_:
            raise ValueError("Cannot invoke event before initializing")

        if self.finalized:
            raise ValueError("Cannot invoke event after finalizing")

        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        if event_type == EventType.LOSS_CALCULATED and (
            "loss" not in kwargs or kwargs["loss"] is None
        ):
            raise ValueError("Loss must be provided for loss calculated event")

        self._check_setup_event_lifecycle(event_type)

        event = None
        mod_data = []
        for event in self.event_lifecycle.events_from_type(event_type):
            if self.state.start_event is None:
                self.state.start_event = event

            for mod in self.modifiers:
                data = mod.update_event(state=self.state, event=event, **kwargs)
                if data is not None:
                    mod_data.append(data)

        assert (
            event is not None
        ), f"Event lifecycle did not return an event for {event_type}"
        self.state.last_event = event
        self.event_called = True

        return mod_data

    def _check_create_state(self, framework: Framework):
        if self.state is not None:
            return

        if framework is None:
            raise ValueError("framework must be provided to create state")

        self.state = State(framework=framework)

    def _check_compile_recipe(self):
        if self.recipe_container.check_compile_recipe():
            self.modifiers = self.recipe_container.compiled_recipe.create_modifier(
                self.state.framework
            )

    def _check_setup_event_lifecycle(self, event_type: EventType):
        if self.event_lifecycle is not None:
            return

        if (
            self.state is None
            or self.state.model is None
            or self.state.start_event is None
            or self.recipe_container.compiled_recipe is None
        ):
            raise ValueError(
                "Cannot invoke event before recipe, model, and start are set"
            )

        if not self.state.sparsification_ready:
            raise ValueError(
                "Cannot invoke event before recipe, model, and start are set"
            )

        for mod in self.modifiers:
            mod.check_initialized()

        if event_type == EventType.BATCH_START:
            self.event_lifecycle = CallbacksEventLifecycle(
                type_first=EventType.BATCH_START, start=self.state.start_event
            )
        elif event_type == EventType.LOSS_CALCULATED:
            self.event_lifecycle = CallbacksEventLifecycle(
                type_first=EventType.LOSS_CALCULATED, start=self.state.start_event
            )
        elif event_type == EventType.OPTIM_PRE_STEP:
            self.event_lifecycle = CallbacksEventLifecycle(
                type_first=EventType.OPTIM_PRE_STEP, start=self.state.start_event
            )
        elif event_type == EventType.OPTIM_POST_STEP:
            self.event_lifecycle = CallbacksEventLifecycle(
                type_first=EventType.OPTIM_POST_STEP, start=self.state.start_event
            )
        else:
            raise ValueError(f"invalid event type {event_type}")

    def _set_model_layer_prefix(self):
        if (
            (compiled_recipe := self.recipe_container.compiled_recipe) is None
            or (metadata := compiled_recipe.metadata) is None
            or (model_metadata := metadata.target_model) is None
        ):
            return False

        self.state.model.layer_prefix = model_metadata.layer_prefix
        return True
