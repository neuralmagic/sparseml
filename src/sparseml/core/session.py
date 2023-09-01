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
from typing import Callable, Any, Union, List, Dict, Tuple
from dataclasses import dataclass

from sparseml.core.state import State
from sparseml.core.event import EventType, Event
from sparseml.core.recipe import Recipe
from sparseml.core.framework import Framework
from sparseml.core.modifier import StageModifiers


__all__ = [
    "SparseSession",
    "create_session",
    "active_session",
    "apply_structure",
    "init",
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

    @property
    def state(self) -> State:
        return self._state

    @property
    def modifiers(self) -> List[StageModifiers]:
        return self._modifiers

    def last_event(self) -> Event:
        return self._state.last_event

    def pre_initialize_structure(
        self,
        model: Any,
        recipe: Union[Recipe, List[Recipe]],
        framework: Framework = None,
        **kwargs,
    ) -> Any:
        self.state.update_framework(framework)
        self.state.update_model(model)
        self.state.update_recipe(recipe)

        self._check_compile_recipe()

        if self._modifiers:
            for modifier in self._modifiers:
                modifier.pre_initialize_structure(state=self.state, **kwargs)

        return self.state.model.model

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
    ) -> Tuple[Any, Any]:
        self.state.update_framework(framework)
        self.state.update_recipe(recipe, recipe_stage, recipe_args)
        self.state.update_model(model)
        self.state.update_optimizer(optimizer, attach_optim_callbacks)
        self.state.update_data(train_data, val_data, test_data, calib_data, copy_data)
        self.state.update_start(start, steps_per_epoch, batches_per_step)

        self._check_compile_recipe()

        if self._modifiers:
            for modifier in self._modifiers:
                modifier.initialize(state=self.state, **kwargs)

        model_return = None
        optim_return = None

        if model:
            model_return = self.state.model.model
        if optimizer:
            optim_return = self.state.optimizer.optimizer

        return model_return, optim_return

    def finalize(self, **kwargs):
        pass

    def apply(self, **kwargs):
        self.initialize(**kwargs)
        self.finalize(**kwargs)

    def apply_structure(
        self, model: Any, recipe: Union[Recipe, List[Recipe]], **kwargs
    ):
        pass

    def event(self, event_type: EventType, **kwargs):
        pass

    def reset(self):
        if self._state:
            del self._state
        self._state = State()

        if self._recipe_modifier:
            del self._recipe_modifier
        self._recipe_modifier = None

    def _check_compile_recipe(self):
        if not self.state.should_recompile_recipe():
            return

        # clear out the modifiers to reinitialize from newly compiled recipe
        if self._modifiers:
            for modifier in self._modifiers:
                if modifier.initialized:
                    modifier.finalize(self.state)
            del self._modifiers

        self.state.recompile_recipe()
        self._modifiers = self.state.compiled_recipe.create_modifiers(
            self.state.framework
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


def apply_structure(**kwargs):
    active_session().apply_structure(**kwargs)


def init(**kwargs):
    active_session().initialize(**kwargs)


def finalize(**kwargs):
    active_session().finalize(**kwargs)


def apply(**kwargs):
    init(**kwargs)
    finalize(**kwargs)


class LifecycleCallbacks:
    @classmethod
    def event(cls, event_type: EventType, **kwargs) -> Any:
        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        return active_session().event(event_type, **kwargs)

    @classmethod
    def batch_start(cls, **kwargs) -> Any:
        return cls.event(EventType.BATCH_START, **kwargs)

    @classmethod
    def batch_end(cls, **kwargs) -> Any:
        return cls.event(EventType.BATCH_END, **kwargs)

    @classmethod
    def optim_stepped(cls, **kwargs) -> Any:
        return cls.event(EventType.OPTIM_POST_STEP, **kwargs)

    @classmethod
    def loss_calculated(cls, **kwargs) -> Any:
        return cls.event(EventType.LOSS_CALCULATED, **kwargs)


callbacks = LifecycleCallbacks
