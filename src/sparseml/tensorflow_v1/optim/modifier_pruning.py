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
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""

import hashlib
from typing import Any, Dict, List, Tuple, Union

from sparseml.sparsification import (
    ConstantPruningModifier as BaseConstantPruningModifier,
)
from sparseml.sparsification import GMPruningModifier as BaseGMPruningModifier
from sparseml.tensorflow_v1.optim.mask_creator_pruning import (
    PruningMaskCreator,
    load_mask_creator,
)
from sparseml.tensorflow_v1.optim.mask_pruning import (
    PruningOpVars,
    apply_op_vars_masks,
    create_ks_scheduled_constant_graph_ops,
    create_summaries_pruning,
    get_or_create_ks_scheduled_graph_ops,
)
from sparseml.tensorflow_v1.optim.modifier import (
    EXTRAS_KEY_SUMMARIES,
    ModifierProp,
    ScheduledModifier,
    ScheduledUpdateModifier,
    TensorFlowModifierYAML,
)
from sparseml.tensorflow_v1.utils import (
    clean_tensor_name,
    get_ops_and_inputs_by_name_or_regex,
    tf_compat,
)
from sparseml.utils import ALL_TOKEN


__all__ = ["ConstantPruningModifier", "GMPruningModifier"]


@TensorFlowModifierYAML()
class ConstantPruningModifier(BaseConstantPruningModifier, ScheduledModifier):
    """
    Holds the sparsity level and shape for a given param constant while training.
    Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantPruningModifier
    |       params: __ALL__
    |       start_epoch: 0.0
    |       end_epoch: 10.0

    :param params: List of str names or regex patterns of names for the parameter
        variables to apply the pruning modifier to. Regex patterns must be specified
        with the prefix 're:'. Can also use the token __ALL__ to specify all
        prunable layers and weights
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1,
        end_epoch: float = -1,
    ):
        super(ConstantPruningModifier, self).__init__(
            params=params,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=None,
        )
        self._prune_op_vars = None
        self._update_ready = None
        self._sparsity = None

    @ModifierProp(serializable=False)
    def ks_group(self) -> str:
        """
        :return: a hashed representation of the settings that identify this instance
        """
        props = self.props(only_serializable=True, format_str=False)
        props = ["{}={}".format(key, val) for key, val in props.items()]
        props.sort()
        props = "&".join(props)

        return "{}".format(hashlib.md5(bytes(props, encoding="utf8")).hexdigest())

    @property
    def prune_op_vars(self) -> Union[None, List[PruningOpVars]]:
        """
        :return: the created pruning op vars in the graph if create_ops has been called,
            else None
        """
        return self._prune_op_vars

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tf_compat.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Tensor,
        graph: tf_compat.Graph,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create the sparsity ops to modify the training graph according to the settings
        for the current instance.

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (list of ops, dict of named ops / tensors)
            to be run or used for modifying the training process.
        """
        mod_ops, mod_extras = super().create_ops(graph, None, None)
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=True)

        params = (
            self.params
            if self.params != ALL_TOKEN
            else [
                clean_tensor_name(var.name)
                for _, var in
                # Have ALL_TOKEN match to all variable names for now
                get_ops_and_inputs_by_name_or_regex(["re:.*"], graph)
            ]
        )

        with graph.as_default():
            update_op, prune_op_vars = create_ks_scheduled_constant_graph_ops(
                graph,
                global_step,
                params,
                start_step,
                end_step,
                self.ks_group,
            )

            mod_extras[EXTRAS_KEY_SUMMARIES] = create_summaries_pruning(prune_op_vars)

        mod_ops.append(update_op)
        self._prune_op_vars = prune_op_vars
        # self._update_ready = tf_compat.constant(False, name="nm_update_ready")

        return mod_ops, mod_extras

    def initialize_session(self, sess: tf_compat.Session):
        """
        Initialize the mask variables for pruning.

        :param sess: the session to use for initializing
        """
        super().initialize_session(sess)
        masks = [op_vars.mask for op_vars in self._prune_op_vars]

        if masks:
            sess.run(tf_compat.variables_initializer(masks))

    def complete_graph(self, graph: tf_compat.Graph, sess: tf_compat.Session):
        """
        Complete modifying the graph.
        Resets the pruned op's variables using the created masks to zero out
        the pruned weights for saving.

        :param graph: the modified graph that should be completed and cleaned.
            if not supplied, then will use the default graph
        :param sess: the session to use for completing the modified graph.
            if not supplied, then will use the default session
        :return: the cleaned graph
        """
        super().complete_graph(graph, sess)

        with graph.as_default():
            apply_op_vars_masks(self.prune_op_vars, self.ks_group, sess)


@TensorFlowModifierYAML()
class GMPruningModifier(BaseGMPruningModifier, ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given variable or variables from
    init_sparsity until final_sparsity is reached over a given amount of time and
    applied with an interpolated function for each step taken.

    Applies based on magnitude pruning without any structure to the pruning.

    | Sample yaml:
    |   !GMPruningModifier
    |       params: __ALL__
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       inter_func: cubic
    |       mask_type: unstructured
    |       leave_enabled: True

    :param params: List of str names or name regex patterns for the variables in the
        graph to apply the pruning modifier to.  Regex patterns must be specified with
        the prefix 're:'.  __ALL__ will match to all parameters.
    :param init_sparsity: The initial sparsity for the variable to
        start with at start_epoch
    :param final_sparsity: The final sparsity for the variable to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to
        update at between start and end
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: The type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        inter_func: str = "cubic",
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        leave_enabled: bool = True,
    ):
        super(GMPruningModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled,
            min_start=-1.0,
            min_end=0.0,
            end_comparator=1,
            min_frequency=-1.0,
        )

        self._mask_creator = mask_type
        if not isinstance(mask_type, PruningMaskCreator):
            self._mask_creator = load_mask_creator(mask_type)
        self._prune_op_vars = None
        self._update_ready = None
        self._sparsity = None
        self._mask_initializer = None

    @ModifierProp(serializable=False)
    def ks_group(self) -> str:
        """
        :return: a hashed representation of the settings that identify this instance
        """
        props = self.props(only_serializable=True, format_str=False)
        props = ["{}={}".format(key, val) for key, val in props.items()]
        props.sort()
        props = "&".join(props)

        return "{}".format(hashlib.md5(bytes(props, encoding="utf8")).hexdigest())

    @ModifierProp(serializable=False)
    def exponent(self) -> float:
        """
        :return: the exponent to be used in for the sparsity schedule
        """

        if self._inter_func == "linear":
            return 1.0

        if self._inter_func == "cubic":
            return 3.0

        if self._inter_func == "inverse_cubic":
            return 1 / 3.0

        raise ValueError(
            "unrecognized value given for inter_func of {}".format(self._inter_func)
        )

    @property
    def prune_op_vars(self) -> Union[None, List[PruningOpVars]]:
        """
        :return: the created pruning op vars in the graph if create_ops has been called,
            else None
        """
        return self._prune_op_vars

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tf_compat.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Tensor,
        graph: tf_compat.Graph,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create the sparsity ops to modify the training graph according to the settings
        for the current instance.

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (list of ops, dict of named ops / tensors)
            to be run or used for modifying the training process.
        """
        mod_ops, mod_extras = super().create_ops(graph, steps_per_epoch, global_step)
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=True)
        update_frequency_step = self.update_frequency_steps(steps_per_epoch)
        params = (
            self._params
            if self._params != ALL_TOKEN
            else [
                clean_tensor_name(var.name)
                for _, var in
                # Have ALL_TOKEN match to all variable names for now
                get_ops_and_inputs_by_name_or_regex(["re:.*"], graph)
            ]
        )

        with graph.as_default():
            (
                update_op,
                prune_op_vars,
                update_ready,
                sparsity,
            ) = get_or_create_ks_scheduled_graph_ops(
                graph,
                global_step,
                params,
                start_step,
                end_step,
                update_frequency_step,
                self._init_sparsity,
                self._final_sparsity,
                self.exponent,
                self._leave_enabled,
                self.ks_group,
                self._mask_creator,
            )

            mod_extras[EXTRAS_KEY_SUMMARIES] = create_summaries_pruning(prune_op_vars)

        mod_ops.append(update_op)
        self._prune_op_vars = prune_op_vars
        self._update_ready = update_ready
        self._sparsity = sparsity

        # Create and cache the mask initializers to be run
        # through initialize_session. When using the estimator,
        # the initialization is done as part of the init_fn of
        # the training scaffold object, at which the graph cannot
        # be changed (hence the creation and caching)
        masks = [op_vars.mask for op_vars in self._prune_op_vars]
        self._mask_initializer = (
            tf_compat.variables_initializer(masks) if masks else None
        )

        return mod_ops, mod_extras

    def initialize_session(self, sess: tf_compat.Session):
        """
        Initialize the mask variables for pruning.

        :param sess: the session to use for initializing
        """
        super().initialize_session(sess)
        if self._mask_initializer:
            sess.run(self._mask_initializer)

    def complete_graph(self, graph: tf_compat.Graph, sess: tf_compat.Session):
        """
        Complete modifying the graph.
        Resets the pruned op's variables using the created masks to zero out
        the pruned weights for saving.

        :param graph: the modified graph that should be completed and cleaned.
            if not supplied, then will use the default graph
        :param sess: the session to use for completing the modified graph.
            if not supplied, then will use the default session
        :return: the cleaned graph
        """
        super().complete_graph(graph, sess)

        with graph.as_default():
            apply_op_vars_masks(self.prune_op_vars, self.ks_group, sess)
