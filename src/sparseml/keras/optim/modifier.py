"""
Contains base code related to modifiers: objects that modify some aspect
of the training process for a model.
For example, learning rate schedules or kernel sparsity (weight pruning)
are implemented as modifiers.
"""

from typing import List, Tuple, Union

import tensorflow as tf
from sparseml.optim import (
    BaseModifier,
    BaseScheduled,
    BaseUpdate,
    ModifierProp,
    ModifierYAML,
)
from sparseml.utils import KERAS_FRAMEWORK


__all__ = [
    "EXTRAS_KEY_LEARNING_RATE",
    "EXTRAS_KEY_SUMMARIES",
    "EXTRAS_KEY_VAR_LIST",
    "NM_RECAL",
    "ModifierProp",
    "KERAS_FRAMEWORK",
    "KerasModifierYAML",
    "Modifier",
    "ScheduledModifier",
    "ScheduledUpdateModifier",
]


EXTRAS_KEY_LEARNING_RATE = "learning_rate"
EXTRAS_KEY_SUMMARIES = "summaries"
EXTRAS_KEY_VAR_LIST = "var_list"

NM_RECAL = "nm_recal"


class KerasModifierYAML(ModifierYAML):
    """
    A decorator to handle making a TensorFlow modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.
    """

    def __init__(self):
        super().__init__(KERAS_FRAMEWORK)


class Modifier(BaseModifier):
    """
    Base modifier class that all TensorFlow modifiers should derive themselves from.
    Handles setting up the expected contracts for modifying graphs, ops, and extras.

    | Modifiers are expected to implement up to 3 different functions for TensorFlow:
    |  - create_ops - inject ops into the graph before the training begins
    |  - create_extras - create extras like learning rate controls before training
    |  - complete_graph - finalize the graph after training has completed
    |
    | Life cycle:
    |   - create model graph
    |   - manager.create_ops()
    |   - manager.create_extras()
    |   - train graph
    |   - manager.complete_graph()
    |   - export graph

    :param log_types: the loggers that can be used by the modifier instance
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

    def __init__(self, log_types: Union[str, List[str]] = None, **kwargs):
        super().__init__(log_types=log_types, **kwargs)
        self.steps_per_epoch = None

    def modify(
        self, model, optimizer, steps_per_epoch: int, input_tensors: tf.Tensor = None
    ):
        callback = None
        return model, optimizer, callback


class ScheduledModifier(Modifier, BaseScheduled):
    """
    The base scheduled update modifier implementation, all scheduled modifiers should
    inherit from this class.
    Offers convenient properties needed for scheduled update modifiers:
    start_epoch, end_epoch

    :param log_types: the loggers that can be used by the modifier instance
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param min_start: The minimum acceptable value for start_epoch, default -1
    :param min_end: The minimum acceptable value for end_epoch, default 0
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be less than, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        log_types: Union[str, List[str]] = None,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        min_start: float = -1.0,
        min_end: float = -1.0,
        end_comparator: Union[int, None] = 0,
        **kwargs,
    ):
        super().__init__(
            log_types=log_types,
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


    | Modifiers are expected to implement up to 3 different functions for TensorFlow:
    |  - create_ops - inject ops into the graph before the training begins
    |  - create_extras - create extras like learning rate controls before training
    |  - complete_graph - finalize the graph after training has completed
    |
    | Life cycle:
    |   - create model graph
    |   - manager.create_ops()
    |   - manager.create_extras()
    |   - train graph
    |   - manager.complete_graph()
    |   - export graph

    :param log_types: the loggers that can be used by the modifier instance
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
        log_types: Union[str, List[str]] = None,
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
            log_types=log_types,
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
