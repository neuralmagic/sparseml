"""
Code dealing with managers of modifiers that modify the training process (graphs, ops, etc)

Expected flow:
    create model graph
    manager.create_ops()
    manager.create_extras()
    train graph
    manager.complete_graph()
    export graph
"""

from typing import List, Tuple, Any

from neuralmagicML.recal import BaseManager
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.recal.modifier import Modifier, ScheduledModifier


__all__ = ["ScheduledModifierManager"]


class ScheduledModifierManager(BaseManager, Modifier):
    """
    The base modifier manager, handles managing multiple ScheduledModifers.

    Modifiers are expected to be able to implement up to 3 different flows to work with tensorflow:
       - create_ops - inject ops into the graph before the training begins
       - create_extras - create extras like learning rate controls before training begins
       - complete_graph - finalize the graph after training has completed
    """

    @staticmethod
    def from_yaml(file_path: str):
        """
        Convenience function used to create the manager of multiple modifiers from a yaml file

        :param file_path: the path to the yaml file to load the modifier from
        :return: ScheduledModifierManager() created from the yaml file
        """
        with open(file_path, "r") as yaml_file:
            yaml_str = yaml_file.read()

        modifiers = Modifier.load_list(yaml_str)
        manager = ScheduledModifierManager(modifiers)

        return manager

    NM_RECAL = "nm_recal"
    NO_OP_UPDATE = "recal_update"

    def __init__(self, modifiers: List[ScheduledModifier]):
        """
        Convenience wrapper around multiple scheduled modifiers

        :param modifiers: the modifiers to wrap
        """
        super().__init__(modifiers=modifiers)

    def create_ops(
        self,
        graph: tf_compat.Graph,
        steps_per_epoch: int,
        global_step: tf_compat.Variable,
    ) -> Tuple[tf_compat.Graph, List[tf_compat.Operation]]:
        """
        Create modifying operations in the graph.
        Returns any ops needed to be run for modifying the training process.
        Additionally returns a modified graph, if not modified returns the original

        :param graph: the graph to be modified
        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :return: a tuple containing the modified graph and extra ops to be run for modifying
        """
        graph, ops = super().create_ops(graph, steps_per_epoch, global_step)

        for mod in self.modifiers:
            graph, mod_ops = mod.create_ops(graph, steps_per_epoch, global_step)

            if mod_ops:
                ops.extend(mod_ops)

        return_ops = []

        with tf_compat.name_scope(ScheduledModifierManager.NM_RECAL):
            with tf_compat.control_dependencies(ops):
                return_ops.append(
                    tf_compat.no_op(ScheduledModifierManager.NO_OP_UPDATE)
                )

        return graph, return_ops

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
        graph, extras = super().create_extras(graph, steps_per_epoch, global_step)

        for mod in self.modifiers:
            graph, mod_extras = mod.create_ops(graph, steps_per_epoch, global_step)

            if mod_extras:
                extras.extend(mod_extras)

        return graph, extras

    def complete_graph(self, graph: tf_compat.GraphDef) -> tf_compat.GraphDef:
        """
        Complete modifying the graph. Should be called after modifying is complete.
        Cleans up any ops that should be removed or reordered.

        :param graph: the modified graph that should be completed and cleaned
        :return: the cleaned graph
        """
        graph = super().complete_graph(graph)

        for mod in self.modifiers:
            graph = mod.complete_graph(graph)

        return graph
