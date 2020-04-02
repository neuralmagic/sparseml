"""
Contains base code related to modifiers: objects that modify some aspect
of the training process for a model.
For example, learning rate schedules or kernel sparsity (weight pruning)
are implemented as modifiers.
"""

from typing import List, Any, Tuple, Union

from neuralmagicML.recal import (
    ModifierProp,
    BaseModifier,
    BaseScheduled,
    BaseUpdate,
    ModifierYAML,
)
from neuralmagicML.utils import TENSORFLOW_FRAMEWORK
from neuralmagicML.tensorflow.utils import tf_compat

__all__ = [
    "ModifierProp",
    "TENSORFLOW_FRAMEWORK",
    "TensorFlowModifierYAML",
    "Modifier",
    "ScheduledModifier",
    "ScheduledUpdateModifier",
]


class TensorFlowModifierYAML(ModifierYAML):
    """
    A decorator to handle making a TensorFlow modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.
    """

    def __init__(self):
        super().__init__(TENSORFLOW_FRAMEWORK)


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
        return Modifier.load_framework_list(yaml_str, TENSORFLOW_FRAMEWORK)

    @staticmethod
    def load_obj(yaml_str: str):
        """
        :param yaml_str:  a string representation of the yaml syntax to
            load a modifier from
        :return: the loaded modifier object
        """
        return Modifier.load_framework_obj(yaml_str, TENSORFLOW_FRAMEWORK)

    def __init__(self, log_types: Union[str, List[str]] = None, **kwargs):
        super().__init__(log_types=log_types, **kwargs)

    def create_ops(
        self,
        graph: tf_compat.Graph,
        steps_per_epoch: int,
        global_step: tf_compat.Variable,
    ) -> Tuple[tf_compat.Graph, List[tf_compat.Operation]]:
        """
        Create modifying operations in the graph.
        Returns any ops needed to be run for modifying the training process.
        Additionally returns a modified graph, if not modified returns the original.

        :param graph: the graph to be modified
        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :return: a tuple containing the modified graph and extra ops to be
            run for modifying
        """
        self._initialized = True

        return graph, []

    def create_extras(
        self,
        graph: tf_compat.Graph,
        steps_per_epoch: int,
        global_step: tf_compat.Variable,
    ) -> Tuple[tf_compat.Graph, List[Tuple[str, Any]]]:
        """
        Create any extras for modifying the training process of a graph.
        These include anything outside of the ops to be run for modifying.

        :param graph: the graph to be modified
        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :return: a tuple containing the modified graph and extras to help modifying
        """
        self._initialized = True

        return graph, []

    def complete_graph(self, graph: tf_compat.GraphDef) -> tf_compat.GraphDef:
        """
        Complete modifying the graph. Should be called after modifying is complete.
        Cleans up any ops that should be removed or reordered.

        :param graph: the modified graph that should be completed and cleaned
        :return: the cleaned graph
        """
        return graph


class ScheduledModifier(Modifier, BaseScheduled):
    """
    The base scheduled update modifier implementation, all scheduled modifiers should
    inherit from this class.
    Offers convenient properties needed for scheduled update modifiers:
    start_epoch, end_epoch


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
        **kwargs
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            min_start=min_start,
            min_end=min_end,
            end_comparator=end_comparator,
            **kwargs
        )


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
        **kwargs
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
            **kwargs
        )


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
