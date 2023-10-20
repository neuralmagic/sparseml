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

from typing import Any, Generator, Tuple

from sparseml.core.event import Event, EventType
from sparseml.core.logger import LoggerManager
from sparseml.core.state import State


__all__ = ["ModelLoggingMixin"]


class ModelLoggingMixin:
    """
    A mixin that adds model level logging functionality
    """

    def log_model_info(self, state: State, event: Event):
        """
        Log model level info to the logger
        Relies on `state.model` having a `loggable_items` method
        that returns a generator of tuples of the loggable item
        name and value. Only logs on BATCH_END type events at the
        end of an epoch. Also relies on `state.loggers` being a
        `LoggerManager` instance.

        :param state: The current state of sparsification
        :param event: The event to update the modifier with
        """

        if not self._should_log_model_info(state, event):
            return
        self._log_epoch(logger_manager=state.loggers, epoch=int(event.current_index))
        self._log_model_loggable_items(
            logger_manager=state.loggers,
            loggable_items=state.model.loggable_items(),
            epoch=event.current_index,
        )

    def _should_log_model_info(self, state: State, event: Event) -> bool:
        """
        Check if we should log model level info
        Criteria:
            - model has a loggable_items method
            - event is of type BATCH_END
            - event is at the end of an epoch
            - state has a logger manager


        :param state: The current state of sparsification
        :param event: The event to update the modifier with
        :return: True if we should log model level info, False otherwise
        """
        return (
            hasattr(state.model, "loggable_items")
            and event.type_ == EventType.BATCH_END
            and isinstance(state.loggers, LoggerManager)
            and event.current_index == int(event.current_index)
        )

    def _log_epoch(self, logger_manager: LoggerManager, epoch: int):
        """
        Log the epoch to the logger_manager

        :param logger_manager: The logger manager to log to
        :param epoch: The epoch to log
        """
        epoch_str = f"Epoch: #{epoch}"
        logger_manager.log_string(tag="Epoch", string=f"{epoch_str:=^20}", step=epoch)

    def _log_model_loggable_items(
        self,
        logger_manager: LoggerManager,
        loggable_items: Generator[Tuple[str, Any], None, None],
        epoch: float,
    ):
        """
        Log the model level loggable items to the logger_manager

        :param logger_manager: The logger manager to log to
        :param loggable_items: The loggable items to log, must be a generator of tuples
            of the loggable item name and value
        :param epoch: The epoch to log
        """
        for loggable_item in loggable_items:
            log_tag, log_value = loggable_item
            if isinstance(log_value, dict):
                logger_manager.log_scalars(tag=log_tag, values=log_value, step=epoch)
            else:
                logger_manager.log_string(tag=log_tag, string=log_value, step=epoch)
