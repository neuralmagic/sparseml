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
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from sparseml.optim import (
    BaseManager,
    BaseScheduled,
    add_framework_metadata,
    load_recipe_yaml_str,
    parse_recipe_variables,
    validate_metadata,
)
from sparseml.tensorflow_v1.optim.modifier import NM_RECAL, Modifier, ScheduledModifier
from sparseml.tensorflow_v1.utils import tf_compat
from sparsezoo.objects import File


__all__ = ["ScheduledModifierManager"]


def _group_modifiers(modifiers: List[ScheduledModifier]) -> List[ScheduledModifier]:
    group_classes = {}  # type: Dict[str, Callable]
    group_mods = {}  # type: Dict[str, List[ScheduledModifier]]
    grouped = []

    for mod in modifiers:
        group = mod.get_group()

        if group:
            if group.__name__ not in group_classes:
                group_classes[group.__name__] = group
                group_mods[group.__name__] = []

            group_mods[group.__name__].append(mod)
        else:
            grouped.append(mod)

    for group, group_const in group_classes.items():
        grouped.append(group_const(group_mods[group]))

    return grouped


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
    def from_yaml(
        file_path: Union[str, File],
        add_modifiers: List[Modifier] = None,
        recipe_variables: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Convenience function used to create the manager of multiple modifiers from a
        recipe file.

        :param file_path: the path to the recipe file to load the modifier from, or
            a SparseZoo model stub to load a recipe for a model stored in SparseZoo.
            SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
            '?recipe_type=<type>' parameter. Can also be a SparseZoo File
            object. i.e. '/path/to/local/recipe.md', 'zoo:model/stub/path',
            'zoo:model/stub/path?recipe_type=transfer'
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the recipe file
        :param recipe_variables: additional variable values to override the recipe
            with (i.e. num_epochs, init_lr)
        :metadata: additional (to the information provided in the recipe) data to be
            preserved and possibly utilized - for reproducibility and completeness
        :return: ScheduledModifierManager() created from the recipe file
        """
        recipe_variables = parse_recipe_variables(recipe_variables)
        yaml_str = load_recipe_yaml_str(file_path, **recipe_variables)
        modifiers = Modifier.load_list(yaml_str)
        if add_modifiers:
            modifiers.extend(add_modifiers)

        validated_metadata = validate_metadata(metadata, yaml_str)

        if metadata is not None:
            validated_metadata = add_framework_metadata(
                validated_metadata, tensorflow_version=tf.__version__
            )

        manager = ScheduledModifierManager(
            modifiers=modifiers, metadata=validated_metadata
        )
        return manager

    RECAL_UPDATE = "recal_update"

    def __init__(
        self,
        modifiers: List[ScheduledModifier],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._orig_modifiers = modifiers
        super().__init__(modifiers=_group_modifiers(modifiers), metadata=metadata)

    def modifiers_to_string_lines(self, modifiers: List[BaseScheduled]) -> List[str]:
        """
        :param modifiers: ignored and overwritten with the original
            (non grouped) modifiers
        :return: a list of lines for a string / yaml representation of the
            modifiers in the manager
        """
        return super().modifiers_to_string_lines(self._orig_modifiers)

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
