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

from sparseml.core.event import EventType
from sparseml.core.framework import Framework
from sparseml.core.lifecycle import SparsificationLifecycle
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
        self._lifecycle = SparsificationLifecycle()

    @property
    def lifecycle(self) -> SparsificationLifecycle:
        return self._lifecycle

    @property
    def state(self) -> State:
        return self._lifecycle.state

    def pre_initialize_structure(
        self,
        model: Any,
        recipe: Union[str, List[str], Recipe, List[Recipe]] = None,
        recipe_stage: Union[str, List[str]] = None,
        recipe_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        framework: Framework = None,
        **kwargs,
    ) -> ModifiedState:
        mod_data = self._lifecycle.pre_initialize_structure(
            model=model,
            recipe=recipe,
            recipe_stage=recipe_stage,
            recipe_args=recipe_args,
            framework=framework,
            **kwargs,
        )

        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=None,
            loss=None,
            modifier_data=mod_data,
        )

    def initialize(
        self,
        framework: Framework = None,
        recipe: Union[str, List[str], "Recipe", List["Recipe"]] = None,
        recipe_stage: Union[str, List[str]] = None,
        recipe_args: Dict[str, Any] = None,
        model: Any = None,
        teacher_model: Any = None,
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
        mod_data = self._lifecycle.initialize(
            framework=framework,
            recipe=recipe,
            recipe_stage=recipe_stage,
            recipe_args=recipe_args,
            model=model,
            teacher_model=teacher_model,
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

        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=self.state.optimizer.optimizer if self.state.optimizer else None,
            loss=self.state.loss.loss if self.state.loss else None,
            modifier_data=mod_data,
        )

    def finalize(self, **kwargs) -> ModifiedState:
        mod_data = self._lifecycle.finalize(**kwargs)

        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=self.state.optimizer.optimizer if self.state.optimizer else None,
            loss=self.state.loss.loss if self.state.loss else None,
            modifier_data=mod_data,
        )

    def apply(self, **kwargs):
        self.initialize(**kwargs)

        return self.finalize(**kwargs)

    def event(
        self, event_type: EventType, batch_data: Any = None, loss: Any = None, **kwargs
    ) -> ModifiedState:
        mod_data = self._lifecycle.event(
            event_type=event_type, batch_data=batch_data, loss=loss, **kwargs
        )

        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=self.state.optimizer.optimizer if self.state.optimizer else None,
            loss=self.state.loss.loss if self.state.loss else None,
            modifier_data=mod_data,
        )

    def reset(self):
        self._lifecycle.reset()


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
    recipe: Union[str, List[str], "Recipe", List["Recipe"]] = None,
    recipe_stage: Union[str, List[str]] = None,
    recipe_args: Dict[str, Any] = None,
    model: Any = None,
    teacher_model: Any = None,
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
        teacher_model=teacher_model,
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
    recipe: Union[str, List[str], "Recipe", List["Recipe"]] = None,
    recipe_stage: Union[str, List[str]] = None,
    recipe_args: Dict[str, Any] = None,
    model: Any = None,
    teacher_model: Any = None,
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
        teacher_model=teacher_model,
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
    def optim_post_step(cls, **kwargs) -> ModifiedState:
        return cls.event(EventType.OPTIM_POST_STEP, **kwargs)

    @classmethod
    def batch_end(cls, **kwargs) -> ModifiedState:
        return cls.event(EventType.BATCH_END, **kwargs)


callbacks = LifecycleCallbacks
