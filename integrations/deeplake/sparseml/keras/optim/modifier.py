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

"""
Contains base code related to modifiers: objects that modify some aspect
of the training process for a model.
For example, learning rate schedules or kernel sparsity (weight pruning)
are implemented as modifiers.
"""

from typing import List, Tuple, Union

from tensorflow import Tensor

from sparseml.keras.utils import KerasLogger, keras
from sparseml.optim import (
    BaseModifier,
    BaseScheduled,
    BaseUpdate,
    ModifierProp,
    ModifierYAML,
)
from sparseml.utils import KERAS_FRAMEWORK


__all__ = [
    "ModifierProp",
    "KerasModifierYAML",
    "Modifier",
    "ModifierProp",
    "ScheduledModifier",
    "ScheduledUpdateModifier",
]


class KerasModifierYAML(ModifierYAML):
    """
    A decorator to handle making a Keras modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.
    """

    def __init__(self):
        super().__init__(KERAS_FRAMEWORK)


class Modifier(BaseModifier):
    """
    Base modifier class that all Keras modifiers should derive themselves from.
    Handles setting up the expected contracts for modifying model and optimizer

    | Modifiers are expected to implement the following functions for Keras:
    |  - modify - modify model and optimizer

    :param kwargs: standard key word args, used to support multi inheritance
    """

    @staticmethod
    def load_list(yaml_str: str):
        """
        :param yaml_str: a string representation of the yaml syntax to
            load modifiers from
        :return: the loaded modifiers list
        """
        return Modifier.load_framework_list(yaml_str, KERAS_FRAMEWORK)

    @staticmethod
    def load_obj(yaml_str: str):
        """
        :param yaml_str:  a string representation of the yaml syntax to
            load a modifier from
        :return: the loaded modifier object
        """
        return Modifier.load_framework_obj(yaml_str, KERAS_FRAMEWORK)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def modify(
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        loggers: Union[KerasLogger, List[KerasLogger]] = None,
        input_tensors: Tensor = None,
    ):
        """
        Modify model, optimizer based on the logic of the modifier. Return the modified
        model, optimizer and a list of callbacks (e.g., to enhance training process)

        :param model: model to modify
        :param optimizer: optimizer to modify
        :param steps_per_epoch: number of steps per epoch
        :param input_tensors: optional input tensor
        :return: model, optimizer, callbacks
        """
        callback = None
        return model, optimizer, callback

    def finalize(self, model: keras.Model):
        """
        Remove extra information related to the modifier from the model that is
        not necessary for exporting

        :param model: a Keras model
        :return: a new Keras model
        """
        return model


class ScheduledModifier(Modifier, BaseScheduled):
    """
    The base scheduled update modifier implementation, all scheduled modifiers should
    inherit from this class.
    Offers convenient properties needed for scheduled update modifiers:
    start_epoch, end_epoch

    | Modifiers are expected to implement the following functions for Keras:
    |  - modify - modify model and optimizer

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be -1, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        min_start: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        **kwargs,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            min_start=min_start,
            min_end=min_end,
            end_comparator=end_comparator,
            **kwargs,
        )

    def start_end_steps(self, steps_per_epoch, after_optim: bool) -> Tuple[int, int]:
        """
        Calculate the start and end steps for this modifier given a certain
        amount of steps per epoch

        :param steps_per_epoch: the number of steps (or batches) taken per epoch
        :param after_optim: True if the start and end are for an operation after
            the optimizer update step has run, False for before
        :return: a tuple containing (the converted start step,
            the converted end step)
        """
        start_step = (
            round(self._start_epoch * steps_per_epoch)
            if self._start_epoch >= 0.0
            else 0
        )
        end_step = (
            round(self._end_epoch * steps_per_epoch) - 1
            if self._end_epoch >= 0.0
            else -1
        )

        if after_optim:
            start_step += 1

            if end_step > -1:
                end_step += 1

        return start_step, end_step


class ScheduledUpdateModifier(ScheduledModifier, BaseUpdate):
    """
    The base scheduled update modifier implementation,
    all scheduled update modifiers should inherit from this class.
    Offers convenient properties needed for scheduled update modifiers: update_frequency

    | Modifiers are expected to implement the following functions for Keras:
    |  - modify - modify model and optimizer

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == -1, then end_epoch can be less than, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param update_frequency: The number of epochs or fraction of epochs to
        update at between start and end
    :param min_frequency: The minimum acceptable value for update_frequency, default -1
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        min_start: float = -1.0,
        min_end: float = -1.0,
        end_comparator: int = 0,
        update_frequency: float = -1.0,
        min_frequency: float = -1.0,
        **kwargs,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            min_start=min_start,
            min_end=min_end,
            end_comparator=end_comparator,
            update_frequency=update_frequency,
            min_frequency=min_frequency,
            **kwargs,
        )

    def update_frequency_steps(self, steps_per_epoch: int) -> int:
        """
        Calculate the update frequency steps for this modifier given a certain
        amount of steps per epoch

        :param steps_per_epoch: the number of steps (or batches) taken per epoch
        :return: a tuple containing (the converted start step,
            the converted end step)
        """
        update_frequency_steps = round(self._update_frequency * steps_per_epoch)

        return update_frequency_steps


def epoch_to_steps(epoch: float, steps_per_epoch: int, min_epoch: float = 0.0) -> int:
    """
    :param epoch: the (fractional) epoch to convert to the proper number of steps
    :param steps_per_epoch: number of steps (batches) taken per epoch while training
    :param min_epoch: if the epoch is less than this, will be set to it. Default 0
    :return: the number of steps representing the epoch and state of the epoch
    """

    if epoch < min_epoch:
        epoch = min_epoch

    return round(steps_per_epoch * epoch)
