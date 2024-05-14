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
from typing import Any, Callable, Dict, List, Optional, Union

from sparseml.core.event import EventType
from sparseml.core.framework import Framework
from sparseml.core.helpers import log_model_info, should_log_model_info
from sparseml.core.lifecycle import SparsificationLifecycle
from sparseml.core.logger import BaseLogger, LoggerManager
from sparseml.core.recipe import Recipe
from sparseml.core.state import ModifiedState, State


__all__ = [
    "SparseSession",
    "create_session",
    "active_session",
    "reset_session",
    "pre_initialize_structure",
    "initialize",
    "finalize",
    "apply",
    "callbacks",
]


@dataclass
class _CallbackContainer:
    """
    A container for a callback and its deregister function

    :param id_: the id of the callback
    :param callback: the callback to invoke
    :param deregister: the function to call to deregister the callback
    :param event_type: the event type the callback is registered for
    :param kwargs: the kwargs the callback was registered with
    """

    id_: int
    callback: Callable
    deregister: Callable
    event_type: EventType
    kwargs: dict


class SparseSession:
    """
    A session for sparsification that holds the lifecycle
    and state for the current sparsification session
    """

    def __init__(self):
        self._lifecycle = SparsificationLifecycle()

    @property
    def lifecycle(self) -> SparsificationLifecycle:
        """
        Lifecycle is used to keep track of where we are in the sparsification
        process and what modifiers are active. It also provides the ability
        to invoke events on the lifecycle.

        :return: the lifecycle for the session
        """
        return self._lifecycle

    @property
    def state(self) -> State:
        """
        State of the current sparsification session. State instance
        is used to store all information such as the recipe, model
        optimizer, data, etc. that is needed for sparsification.

        :return: the current state of the session
        """
        return self._lifecycle.state

    def pre_initialize_structure(
        self,
        model: Any,
        recipe: Union[str, List[str], Recipe, List[Recipe], None] = None,
        recipe_stage: Union[str, List[str], None] = None,
        recipe_args: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        framework: Optional[Framework] = None,
        **kwargs,
    ) -> ModifiedState:
        """
        A method to pre-initialize the structure of the model for sparsification.
        This will run the pre-initialize structure method for each modifier in the
        session's lifecycle. This will also set the session's state to the
        pre-initialized state. Takes care of cases when the model(s) structure
        has been previously modified by a modifier.

        :param model: the model to pre-initialize the structure for
        :param recipe: the recipe to use for the sparsification, can be a path to a
            recipe file, a raw recipe string, a recipe object, or a list
            of recipe objects.
        :param recipe_stage: the stage to use for the sparsification
        :param recipe_args: the args to use for overriding the recipe defaults
        :param framework: the framework to use for the sparsification
        :return: A ModifiedState instance holding the modified model and modifier_data
            after pre-initializing the structure
        """
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
        framework: Optional[Framework] = None,
        recipe: Union[str, List[str], "Recipe", List["Recipe"], None] = None,
        recipe_stage: Union[str, List[str], None] = None,
        recipe_args: Union[Dict[str, Any], None] = None,
        model: Optional[Any] = None,
        teacher_model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        attach_optim_callbacks: bool = True,
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        calib_data: Optional[Any] = None,
        copy_data: bool = True,
        start: Optional[float] = None,
        steps_per_epoch: Optional[int] = None,
        batches_per_step: Optional[int] = None,
        loggers: Union[None, LoggerManager, List[BaseLogger]] = None,
        **kwargs,
    ) -> ModifiedState:
        """
        Initialize the session for sparsification. This will run the initialize method
        for each modifier in the session's lifecycle. This will also set the session's
        state to the initialized state.

        :param framework: the framework to use for the sparsification
        :param recipe: the recipe to use for the sparsification, can be a path to a
            recipe file, a raw recipe string, a recipe object, or a list
            of recipe objects.
        :param recipe_stage: the stage to target for the sparsification
        :param recipe_args: the args to use for overriding the recipe defaults
        :param model: the model to sparsify
        :param teacher_model: the teacher model to use for knowledge distillation
        :param optimizer: the optimizer to use for the sparsification
        :param attach_optim_callbacks: True to attach the optimizer callbacks to the
            sparsification lifecycle, False otherwise
        :param train_data: the training data to use for the sparsification
        :param val_data: the validation data to use for the sparsification
        :param test_data: the testing data to use for the sparsification
        :param calib_data: the calibration data to use for the sparsification
        :param copy_data: True to copy the data, False otherwise
        :param start: the start epoch to use for the sparsification
        :param steps_per_epoch: the number of steps per epoch to use for the
            sparsification
        :param batches_per_step: the number of batches per step to use for
            sparsification
        :param loggers: the logger manager to setup logging important info
            and milestones to, also accepts a list of BaseLogger(s)
        :param kwargs: additional kwargs to pass to the lifecycle's initialize method
        :return: the modified state of the session after initializing
        """

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
            loggers=loggers,
            **kwargs,
        )

        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=self.state.optimizer.optimizer if self.state.optimizer else None,
            loss=self.state.loss,
            modifier_data=mod_data,
        )

    def finalize(self, **kwargs) -> ModifiedState:
        """
        Finalize the session for sparsification. This will run the finalize method
        for each modifier in the session's lifecycle. This will also set the session's
        state to the finalized state.

        :param kwargs: additional kwargs to pass to the lifecycle's finalize method
        :return: the modified state of the session after finalizing
        """
        mod_data = self._lifecycle.finalize(**kwargs)

        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=self.state.optimizer.optimizer if self.state.optimizer else None,
            loss=self.state.loss,
            modifier_data=mod_data,
        )

    def apply(self, **kwargs):
        """
        Apply the recipe in one-shot manner. This will invoke the initialize
        and then finalize methods for each modifier in the session's lifecycle.
        This will also set the session's state to the finalized state.

        :param kwargs: additional kwargs to pass to the lifecycle's initialize and
            finalize methods
        """
        self.initialize(**kwargs)

        return self.finalize(**kwargs)

    def event(
        self,
        event_type: EventType,
        batch_data: Optional[Any] = None,
        loss: Optional[Any] = None,
        **kwargs,
    ) -> ModifiedState:
        """
        Invoke an event for current SparseSession.

        :param event_type: the event type to invoke
        :param batch_data: the batch data to use for the event
        :param loss: the loss to use for the event if any
        :param kwargs: additional kwargs to pass to the lifecycle's event method
        :return: the modified state of the session after invoking the event
        """
        mod_data = self._lifecycle.event(
            event_type=event_type, batch_data=batch_data, loss=loss, **kwargs
        )
        return ModifiedState(
            model=self.state.model.model if self.state.model else None,
            optimizer=self.state.optimizer.optimizer if self.state.optimizer else None,
            loss=self.state.loss,  # TODO: is this supposed to be a different type?
            modifier_data=mod_data,
        )

    def log(self, event_type: EventType, loss: Optional[Any] = None):
        """
        Log model and loss information for the current event type

        :param event_type: the event type to log for
        :param loss: the loss to log if any
        """
        self._log_model_info()
        self._log_loss(event_type=event_type, loss=loss)

    def reset(self):
        """
        Reset the session to its initial state
        """
        self._lifecycle.reset()

    def reset_stage(self):
        """
        Reset the session for starting a new stage, recipe and model stays intact
        """
        self.lifecycle.initialized_ = False
        self.lifecycle.finalized = False

    def get_serialized_recipe(self) -> str:
        """
        :return: serialized string of the current compiled recipe
        """
        recipe = self.lifecycle.recipe_container.compiled_recipe
        return recipe.yaml()

    def _log_model_info(self):
        # Log model level logs if cadence reached
        event_lifecycle = self._lifecycle.event_lifecycle
        if event_lifecycle is None:
            # event lifecycle not available
            # when recipe is not provided
            return

        epoch = event_lifecycle.current_index

        if (
            should_log_model_info(
                model=self.state.model,
                loggers=self.state.loggers,
                current_log_step=epoch,
                last_log_step=self.state._last_log_step,
            )
            and self.state.loggers.frequency_manager.is_epoch_frequency_manager
        ):
            log_model_info(
                state=self.state,
                current_log_step=epoch,
            )
            # update last log epoch
            self.state.loggers.log_written(epoch)

    def _log_loss(self, event_type: EventType, loss: Any):
        if event_type != EventType.LOSS_CALCULATED:
            # only log loss when loss is calculated
            return
        event_lifecycle = self._lifecycle.event_lifecycle

        if event_lifecycle is None:
            # event lifecycle not available
            # when recipe is not provided
            return

        epoch = event_lifecycle.current_index
        if self.state.loggers.frequency_manager.is_optim_frequency_manager:
            # log integer step for optimizer frequency manager
            current_step = int(
                self.state.loggers.epoch_to_step(
                    epoch=epoch,
                    steps_per_epoch=len(self.state.data.train),
                )
            )
        else:
            # log float epoch for epoch frequency manager
            current_step = epoch

        # always log loss if available
        if loss is not None:
            loss = loss if isinstance(loss, dict) else {"loss": loss}
            self.state.loggers.metric.log_scalars(
                tag="Loss", values=loss, step=current_step
            )


_global_session = SparseSession()
_local_storage = threading.local()
_local_storage.session = _global_session


@contextmanager
def create_session() -> SparseSession:
    """
    Context manager to create and yield a new session for sparsification.
    This will set the active session to the new session for the duration
    of the context.

    :return: the new session
    """
    global _local_storage
    orig_session = getattr(_local_storage, "session", None)
    new_session = SparseSession()
    _local_storage.session = new_session
    try:
        yield new_session
    finally:
        _local_storage.session = orig_session


def active_session() -> SparseSession:
    """
    :return: the active session for sparsification
    """
    global _local_storage
    return getattr(_local_storage, "session", _global_session)


def reset_session():
    """
    Reset the currently active session to its initial state
    """
    session = active_session()
    session._lifecycle.reset()


def pre_initialize_structure(**kwargs):
    """
    A method to pre-initialize the structure of the model for the active session

    :param kwargs: the kwargs to pass to the active session's pre-initialize-structure
        method
    """
    active_session().pre_initialize_structure(**kwargs)


def initialize(
    framework: Optional[Framework] = None,
    recipe: Union[str, List[str], "Recipe", List["Recipe"], None] = None,
    recipe_stage: Union[str, List[str], None] = None,
    recipe_args: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    teacher_model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    attach_optim_callbacks: bool = True,
    train_data: Optional[Any] = None,
    val_data: Optional[Any] = None,
    test_data: Optional[Any] = None,
    calib_data: Optional[Any] = None,
    copy_data: bool = True,
    start: Optional[float] = None,
    steps_per_epoch: Optional[int] = None,
    batches_per_step: Optional[int] = None,
    **kwargs,
) -> ModifiedState:
    """
    A method to initialize the active session for sparsification

    :param framework: the framework to use for the sparsification
    :param recipe: the recipe to use for the sparsification, can be a path to a
        recipe file, a raw recipe string, a recipe object, or a list of recipe objects.
    :param recipe_stage: the stage to target for the sparsification
    :param recipe_args: the args to use for overriding the recipe defaults
    :param model: the model to sparsify
    :param teacher_model: the teacher model to use for knowledge distillation
    :param optimizer: the optimizer to use for the sparsification
    :param attach_optim_callbacks: True to attach the optimizer callbacks to the
        sparsification lifecycle, False otherwise
    :param train_data: the training data to use for the sparsification
    :param val_data: the validation data to use for the sparsification
    :param test_data: the testing data to use for the sparsification
    :param calib_data: the calibration data to use for the sparsification
    :param copy_data: True to copy the data, False otherwise
    :param start: the start epoch to use for the sparsification
    :param steps_per_epoch: the number of steps per epoch to use for the
        sparsification
    :param batches_per_step: the number of batches per step to use for
        sparsification
    :param kwargs: additional kwargs to pass to the lifecycle's initialize method
    :return: the modified state of the active session after initializing
    """
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
    """
    Method to finalize the active session for sparsification

    :param kwargs: additional kwargs to pass to the lifecycle's finalize method
    :return: the modified state of the active session after finalizing
    """
    return active_session().finalize(**kwargs)


def apply(
    framework: Optional[Framework] = None,
    recipe: Union[str, List[str], "Recipe", List["Recipe"], None] = None,
    recipe_stage: Union[str, List[str], None] = None,
    recipe_args: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    teacher_model: Optional[Any] = None,
    train_data: Optional[Any] = None,
    val_data: Optional[Any] = None,
    test_data: Optional[Any] = None,
    calib_data: Optional[Any] = None,
    copy_data: bool = True,
    start: Optional[float] = None,
    steps_per_epoch: Optional[int] = None,
    batches_per_step: Optional[int] = None,
    **kwargs,
) -> ModifiedState:
    """
    A method to apply the recipe in one-shot manner. This will invoke the initialize
    and then finalize methods for each modifier in the active session's lifecycle.

    :param framework: the framework to use for the sparsification
    :param recipe: the recipe to use for the sparsification, can be a path to a
        recipe file, a raw recipe string, a recipe object, or a list of recipe objects.
    :param recipe_stage: the stage to target for the sparsification
    :param recipe_args: the args to use for overriding the recipe defaults
    :param model: the model to sparsify
    :param teacher_model: the teacher model to use for knowledge distillation
    :param train_data: the training data to use for the sparsification
    :param val_data: the validation data to use for the sparsification
    :param test_data: the testing data to use for the sparsification
    :param calib_data: the calibration data to use for the sparsification
    :param copy_data: True to copy the data, False otherwise
    :param start: the start epoch to use for the sparsification
    :param steps_per_epoch: the number of steps per epoch to use for the
        sparsification
    :param batches_per_step: the number of batches per step to use for
    :param kwargs: additional kwargs to pass to the current session's apply method
    :return: the modified state of the active session after applying the recipe
    """
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
    """
    A class for invoking lifecycle events for the active session
    """

    @classmethod
    def event(cls, event_type: EventType, **kwargs) -> ModifiedState:
        """
        Invoke an event for the active session

        :param event_type: the event type to invoke
        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        if event_type in [EventType.PRE_INIT, EventType.INITIALIZE, EventType.FINALIZE]:
            raise ValueError(
                f"Cannot invoke {event_type} event. "
                f"Use the corresponding method instead."
            )

        # skip event callbacks if no recipe was provided
        if not active_session().lifecycle.recipe_container.check_any_recipe_exists():
            return

        return active_session().event(event_type, **kwargs)

    @classmethod
    def batch_start(cls, batch_data: Optional[Any] = None, **kwargs) -> ModifiedState:
        """
        Invoke a batch start event for the active session

        :param batch_data: the batch data to use for the event
        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        return cls.event(EventType.BATCH_START, batch_data=batch_data, **kwargs)

    @classmethod
    def loss_calculated(cls, loss: Optional[Any] = None, **kwargs) -> ModifiedState:
        """
        Invoke a loss calculated event for the active session

        :param loss: the loss to use for the event
        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        # log loss if loss calculated
        active_session()._log_loss(event_type=EventType.LOSS_CALCULATED, loss=loss)
        return cls.event(EventType.LOSS_CALCULATED, loss=loss, **kwargs)

    @classmethod
    def optim_pre_step(cls, **kwargs) -> ModifiedState:
        """
        Invoke an optimizer pre-step event for the active session

        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        return cls.event(EventType.OPTIM_PRE_STEP, **kwargs)

    @classmethod
    def optim_post_step(cls, **kwargs) -> ModifiedState:
        """
        Invoke an optimizer post-step event for the active session

        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        return cls.event(EventType.OPTIM_POST_STEP, **kwargs)

    @classmethod
    def batch_end(cls, **kwargs) -> ModifiedState:
        """
        Invoke a batch end event for the active session

        :param kwargs: additional kwargs to pass to the current session's event method
        :return: the modified state of the active session after invoking the event
        """
        active_session()._log_model_info()
        return cls.event(EventType.BATCH_END, **kwargs)


callbacks = LifecycleCallbacks
