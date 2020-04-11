"""
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

from typing import List, Any, Union, Dict, Tuple

from neuralmagicML.recal import BaseManager
from neuralmagicML.utils import clean_path, create_parent_dirs
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.recal.modifier import Modifier, ScheduledModifier


__all__ = ["ScheduledModifierManager"]


class ScheduledModifierManager(BaseManager, Modifier):
    """
    The base modifier manager, handles managing multiple ScheduledModifier.

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

    :param modifiers: the modifiers to wrap
    """

    @staticmethod
    def from_yaml(file_path: str):
        """
        Convenience function used to create the manager of multiple modifiers
        from a yaml file.

        :param file_path: the path to the yaml file to load the modifier from
        :return: ScheduledModifierManager() created from the yaml file
        """
        with open(file_path, "r") as yaml_file:
            yaml_str = yaml_file.read()

        modifiers = Modifier.load_list(yaml_str)
        manager = ScheduledModifierManager(modifiers)

        return manager

    NM_RECAL = "nm_recal"
    RECAL_UPDATE = "recal_update"

    def __init__(self, modifiers: List[ScheduledModifier]):
        super().__init__(modifiers=modifiers)

    def save(self, file_path: str):
        """
        :param file_path: the file path to save the yaml config representation to
        """
        file_path = clean_path(file_path)
        create_parent_dirs(file_path)

        with open(file_path, "w") as yaml_file:
            yaml_file.write(Modifier.list_to_yaml(self.modifiers))

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Tensor = None,
        graph: tf_compat.Graph = None,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create modifying operations and tensors in the graph.

        | Returns a tuple containing:
        |   - modifying ops that should be run in a session on each global step.
        |   - named extras (ops / tensors) created in the graph that can be used
        |     by other ops such as a learning rate for the optimizer

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training.
            if not supplied, then will use get_or_create_global_step()
        :param graph: the graph to be modified,
            if not supplied, then will use the default graph
        :return: a tuple (list of ops, dict of named ops / tensors)
            to be run or used for modifying the training process
        """
        if not graph:
            graph = tf_compat.get_default_graph()

        if not global_step:
            with graph.as_default():
                global_step = tf_compat.train.get_or_create_global_step()

        mod_ops, mod_extras = super().create_ops(steps_per_epoch, global_step, graph)
        tmp_ops = []

        for mod in self.modifiers:
            ops, extras = mod.create_ops(steps_per_epoch, global_step, graph)

            if ops:
                tmp_ops.extend(ops)

            if extras:
                for key, val in extras.items():
                    if key not in mod_extras:
                        mod_extras[key] = []

                    if isinstance(val, List):
                        mod_extras[key].extend(val)
                    else:
                        mod_extras[key].append(val)

        with tf_compat.name_scope(ScheduledModifierManager.NM_RECAL):
            mod_ops.append(
                tf_compat.group(tmp_ops, name=ScheduledModifierManager.RECAL_UPDATE)
            )

        return mod_ops, mod_extras

    def initialize_session(self, sess: tf_compat.Session = None):
        """
        Initialize any state for a session such as variables.
        This is an optional call, only needed if global_variables_initializer
        is not used.

        :param sess: the session to use for initializing
        """
        if not sess:
            sess = tf_compat.get_default_session()

        super().initialize_session(sess)

        for mod in self.modifiers:
            mod.initialize_session(sess)

    def complete_graph(
        self, graph: tf_compat.Graph = None, sess: tf_compat.Session = None
    ):
        """
        Complete modifying the graph. Should be called after modifying is complete.
        Cleans up any ops that should be removed or reordered.

        :param graph: the modified graph that should be completed and cleaned.
            if not supplied, then will use the default graph
        :param sess: the session to use for completing the modified graph.
            if not supplied, then will use the default session
        :return: the cleaned graph
        """
        super().complete_graph(graph, sess)

        if not graph:
            graph = tf_compat.get_default_graph()

        if not sess:
            sess = tf_compat.get_default_session()

        for mod in self.modifiers:
            mod.complete_graph(graph, sess)
