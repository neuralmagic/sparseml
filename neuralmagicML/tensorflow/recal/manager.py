"""
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

import collections
import itertools
from typing import List, Any, Union, Dict, Tuple

from neuralmagicML.recal import BaseManager
from neuralmagicML.utils import clean_path, create_parent_dirs
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.recal.modifier import (
    NM_RECAL,
    Modifier,
    ScheduledModifier,
)

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

    RECAL_UPDATE = "recal_update"

    def _group_modifiers(
        self, modifiers: List[ScheduledModifier]
    ) -> List[ScheduledModifier]:
        # List of individial modifiers which stay separately
        self._non_group_mods = []

        # Each set of modifiers in a group is cached based on the group class name
        # The resulting container modifier will be cached as the first element of
        # the 2-element list below. We need it in order to restore individial
        # modifier's log type before saving the modifiers
        self._group_mods = collections.defaultdict(lambda: [None, []])

        for mod in modifiers:
            group_cls = mod.get_group()
            if group_cls is None:
                self._non_group_mods.append(mod)
            else:
                # Add this modifier into the list. The container modifier
                # is added later into the first slot.
                self._group_mods[group_cls.__name__][1].append(mod)
        res = self._non_group_mods
        for group_cls_name, (_, mod_list) in self._group_mods.items():
            constructor = mod_list[0].get_group()
            new_group_mod = constructor(mod_list)
            # Add the resulting container modifier into the slot for caching
            self._group_mods[group_cls_name][0] = new_group_mod
            res.append(new_group_mod)
        return res

    def __init__(self, modifiers: List[ScheduledModifier]):
        self._non_group_mods = None
        self._group_mods = None
        grouped_modifiers = self._group_modifiers(modifiers)
        super().__init__(modifiers=grouped_modifiers)

    def save(self, file_path: str):
        """
        :param file_path: the file path to save the yaml config representation to
        """
        file_path = clean_path(file_path)
        create_parent_dirs(file_path)

        # Recover the original modifiers before saving
        orig_mods = self._non_group_mods
        for group_cls_name, (grouped_mod, mod_list) in self._group_mods.items():
            for mod in mod_list:
                # Force individual modifier's log types to be that of the
                # container modifier
                mod.log_types = grouped_mod.log_types
                orig_mods.append(mod)
        with open(file_path, "w") as yaml_file:
            yaml_file.write(Modifier.list_to_yaml(orig_mods))

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

        ops, extras = super().create_ops(steps_per_epoch, global_step, graph)

        mod_ops_extras = [
            mod.create_ops(steps_per_epoch, global_step, graph)
            for mod in self.modifiers
        ]  # List[Tuple[List, Dict]]

        merged_ops = list(
            itertools.chain.from_iterable(
                [_ops for (_ops, _) in mod_ops_extras if _ops]
            )
        )

        mod_extras = [
            _extras for (_, _extras) in mod_ops_extras
        ]  # List[Dict[str, Any]]

        extras = {}
        for _extras in mod_extras:
            for key, val in _extras.items():
                if key not in extras:
                    extras[key] = val
                else:
                    # This key exists before either as a list or a single value
                    if not isinstance(extras[key], List):
                        raise ValueError(
                            "extras[{}] has been recorded with unique "
                            "value and cannot be merged".format(key)
                        )
                    if not isinstance(val, List):
                        raise ValueError(
                            "extras[{}] has been recorded as list, "
                            "requiring new list to merge".format(key)
                        )
                    extras[key].extend(val)

        with tf_compat.name_scope(NM_RECAL):
            ops.append(
                tf_compat.group(merged_ops, name=ScheduledModifierManager.RECAL_UPDATE)
            )

        return ops, extras

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

        return graph
