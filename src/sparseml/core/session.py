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

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union

from sparseml.core.event import (
    CallbacksEventLifecycle,
    EventType,
    WrappedOptimEventLifecycle,
)
from sparseml.core.framework import Framework
from sparseml.core.modifier import StageModifiers
from sparseml.core.recipe import Recipe
from sparseml.core.state import ModifiedState, State


__all__ = [
    "SparseSession",
    "create_session",
    "active_session",
    "pre_initialize_structure",
    "initialize",
    "finalize",
    "apply",
    "callbacks",
]


@dataclass
class _CallbackContainer:
    id_: int
    callback: Callable
    deregister: Callable
    event_type: EventType
    kwargs: dict


class SparseSession:
    def __init__(self):
        self._state: State = State()
        self._modifiers: List[StageModifiers] = []
        self._initialized_structure = False
        self._initialized = False
        self._finalized = False
        self._event_called = False

    @property
    def state(self) -> State:
        return self._state

    @property
    def modifiers(self) -> List[StageModifiers]:
        return self._modifiers

    @property
    def initialized_structure(self) -> bool:
        return self._initialized_structure

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def event_called(self) -> bool:
        return self._event_called

    def pre_initialize_structure(
        self,
        model: Any,
        recipe: Union[Recipe, List[Recipe]],
        framework: Framework = None,
        **kwargs,
    ) -> ModifiedState:
        self.state.update_framework(framework)
        self.state.update_model(model)
        self.state.update_recipe(recipe)

        self._check_compile_recipe()
        modifier_data = []

        for modifier in self._modifiers:
            data = modifier.pre_initialize_structure(state=self.state, **kwargs)
            if data:
                modifier_data.append(data)

        self._initialized_structure = True

        return ModifiedState(
            model=self.state.model.model,
            optimizer=None,
            loss=None,
            modifier_data=modifier_data,
        )

    def initialize(
        self,
        framework: Framework = None,
        recipe: Union[str, List[str], Recipe, List[Recipe]] = None,
        recipe_stage: str = None,
        recipe_args: Dict[str, Any] = None,
        model: Any = None,
        optimizer: Any = None,
        attach_optim_callbacks: bool = True,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        calib_data: Any = None,
        copy_data: bool = True,
        start: float = None,
        steps_per_epoch: int = None,
        batches_per_step: int = None,
        **kwargs,
    ) -> ModifiedState:
        if self.event_called:
            raise ValueError("Cannot initialize after invoking an event")

        if self.finalized:
            raise ValueError("Cannot initialize after finalizing")

        self.state.update_framework(framework)
        self.state.update_recipe(recipe, recipe_stage, recipe_args)
        self.state.update_model(model)
        self.state.update_optimizer(optimizer, attach_optim_callbacks)
        self.state.update_data(train_data, val_data, test_data, calib_data, copy_data)
        self.state.update_start(start, steps_per_epoch, batches_per_step)

        self._check_compile_recipe()
        modifier_data = []

        if self._modifiers:
            for modifier in self._modifiers:
                data = modifier.initialize(state=self.state, **kwargs)
                if data:
                    modifier_data.append(data)

        self._initialized = True

        return ModifiedState(
            model=self.state.model.model,
            optimizer=self.state.optimizer.optimizer,
            loss=self.state.loss.loss,
            modifier_data=modifier_data,
        )

    def finalize(self, **kwargs) -> ModifiedState:
        if not self.initialized:
            raise ValueError("Cannot finalize before initializing")

        if self.finalized:
            raise ValueError("Cannot finalize more than once")

        modifier_data = []

        for modifier in self._modifiers:
            data = modifier.finalize(state=self.state, **kwargs)
            if data:
                modifier_data.append(data)

        self._finalized = True

        return ModifiedState(
            model=self.state.model.model,
            optimizer=self.state.optimizer.optimizer,
            loss=self.state.loss.loss,
            modifier_data=modifier_data,
        )

    def apply(self, **kwargs):
        self.initialize(**kwargs)

        return self.finalize(**kwargs)

    def event(
        self, event_type: EventType, batch_data: Any = None, loss: Any = None, **kwargs
    ) -> ModifiedState:
        if not self.initialized:
            raise ValueError("Cannot invoke event before initializing")

        if self.finalized:
            raise ValueError("Cannot invoke event after finalizing")

        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        if event_type == EventType.LOSS_CALCULATED and loss is None:
            raise ValueError("Loss must be provided for loss calculated event")

        self._check_setup_lifecycle(event_type)

        event = None
        modifier_data = []
        for event in self.state.event_lifecycle.events_from_type(event_type):
            for modifier in self._modifiers:
                data = modifier.update_event(
                    state=self.state,
                    event=event,
                    batch_data=batch_data,
                    loss=loss,
                    **kwargs,
                )
                if data:
                    modifier_data.append(data)

        assert event is not None, f"No events generated for event type {event_type}"
        self.state.last_event = event
        self._event_called = True

        return ModifiedState(
            model=self.state.model.model,
            optimizer=self.state.optimizer.optimizer,
            loss=self.state.loss.loss,
            modifier_data=modifier_data,
        )

    def reset(self):
        if self._state:
            del self._state
        self._state = State()

        if self._modifiers:
            if self.initialized and not self.finalized:
                for modifier in self._modifiers:
                    modifier.finalize(self.state)

            del self._modifiers

        self._modifiers = []
        self._initialized_structure = False
        self._initialized = False
        self._finalized = False
        self._event_called = False

    def _check_compile_recipe(self):
        if not self.state.recipe_changed and self._modifiers is not None:
            # recipe hasn't changed and modifiers set, no need to recompile
            return

        if self.state.recipes is None:
            # no recipes currently, return
            return

        if self.state.recipe_changed:
            self.state.recompile_recipe()

            if self._modifiers:
                # clear out the modifiers to reinitialize from newly compiled recipe
                for modifier in self._modifiers:
                    if modifier._initialized:
                        modifier.finalize(self.state)
                del self._modifiers

        if self.state.recipe_modifier_ready:
            self._modifiers = self.state.compiled_recipe.create_modifier(
                self.state.framework
            )

    def _check_setup_lifecycle(self, event_type: EventType):
        if self.state.event_lifecycle is not None:
            return

        # first event call, setup lifecycle and make sure everything is initialized
        if not self.state.recipe_modifier_ready:
            raise ValueError(
                "Cannot invoke event before recipe, model, and start are set"
            )

        for modifier in self._modifiers:
            modifier.check_initialized()

        if event_type == EventType.BATCH_START:
            # utilizing callbacks pathway, ensure optim is not wrapped
            if self.state.optim_wrapped:
                raise ValueError(
                    "Cannot use batch callbacks with wrapped optimizer, "
                    "set attach_optim_callbacks to False when initializing "
                )
            self.state.event_lifecycle = CallbacksEventLifecycle(
                event_type, self.state.start_event
            )
        elif self.state.optim_wrapped:
            # utilizing wrapped optimizer for callbacks
            self.state.event_lifecycle = WrappedOptimEventLifecycle(
                event_type, self.state.start_event
            )
        else:
            raise ValueError(
                "First event must be batch_start or "
                "attach_optim_callbacks must be True"
            )


_global_session = SparseSession()
_local_storage = threading.local()
_local_storage.session = _global_session


@contextmanager
def create_session() -> SparseSession:
    global _local_storage
    orig_session = getattr(_local_storage, "session", None)
    new_session = SparseSession()
    _local_storage.session = new_session
    try:
        yield new_session
    finally:
        _local_storage.session = orig_session


def active_session() -> SparseSession:
    global _local_storage
    return getattr(_local_storage, "session", _global_session)


def pre_initialize_structure(**kwargs):
    active_session().pre_initialize_structure(**kwargs)


def initialize(
    framework: Framework = None,
    recipe: Union[str, List[str], Recipe, List[Recipe]] = None,
    recipe_stage: str = None,
    recipe_args: Dict[str, Any] = None,
    model: Any = None,
    optimizer: Any = None,
    attach_optim_callbacks: bool = True,
    train_data: Any = None,
    val_data: Any = None,
    test_data: Any = None,
    calib_data: Any = None,
    copy_data: bool = True,
    start: float = None,
    steps_per_epoch: int = None,
    batches_per_step: int = None,
    **kwargs,
) -> ModifiedState:
    return active_session().initialize(
        framework=framework,
        recipe=recipe,
        recipe_stage=recipe_stage,
        recipe_args=recipe_args,
        model=model,
        optimizer=optimizer,
        attach_optim_callbacks=attach_optim_callbacks,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        calib_data=calib_data,
        copy_data=copy_data,
        start=start,
        steps_per_epoch=steps_per_epoch,
        batches_per_step=batches_per_step,
        **kwargs,
    )


def finalize(**kwargs) -> ModifiedState:
    return active_session().finalize(**kwargs)


def apply(
    framework: Framework = None,
    recipe: Union[str, List[str], Recipe, List[Recipe]] = None,
    recipe_stage: str = None,
    recipe_args: Dict[str, Any] = None,
    model: Any = None,
    train_data: Any = None,
    val_data: Any = None,
    test_data: Any = None,
    calib_data: Any = None,
    copy_data: bool = True,
    start: float = None,
    steps_per_epoch: int = None,
    batches_per_step: int = None,
    **kwargs,
) -> ModifiedState:
    return active_session().apply(
        framework=framework,
        recipe=recipe,
        recipe_stage=recipe_stage,
        recipe_args=recipe_args,
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        calib_data=calib_data,
        copy_data=copy_data,
        start=start,
        steps_per_epoch=steps_per_epoch,
        batches_per_step=batches_per_step,
        **kwargs,
    )


class LifecycleCallbacks:
    @classmethod
    def event(cls, event_type: EventType, **kwargs) -> ModifiedState:
        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        return active_session().event(event_type, **kwargs)

    @classmethod
    def batch_start(cls, batch_data: Any = None, **kwargs) -> ModifiedState:
        return cls.event(EventType.BATCH_START, batch_data=batch_data, **kwargs)

    @classmethod
    def loss_calculated(cls, loss: Any = None, **kwargs) -> ModifiedState:
        return cls.event(EventType.LOSS_CALCULATED, loss=loss, **kwargs)

    @classmethod
    def optim_pre_step(cls, **kwargs) -> ModifiedState:
        return cls.event(EventType.OPTIM_PRE_STEP, **kwargs)

    @classmethod
    def optim_stepped(cls, **kwargs) -> ModifiedState:
        return cls.event(EventType.OPTIM_POST_STEP, **kwargs)

    @classmethod
    def batch_end(cls, **kwargs) -> ModifiedState:
        return cls.event(EventType.BATCH_END, **kwargs)


callbacks = LifecycleCallbacks
