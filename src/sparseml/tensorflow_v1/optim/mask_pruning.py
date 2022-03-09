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
Code related to applying a mask onto a variable to impose kernel sparsity,
aka model pruning, on a TensorFlow graph.
"""

from collections import namedtuple
from typing import List, Tuple


try:
    import tensorflow.contrib.graph_editor as graph_editor

    tf_contrib_err = None
except Exception as err:
    graph_editor = None
    tf_contrib_err = err

from sparseml.tensorflow_v1.optim.mask_creator_pruning import PruningMaskCreator
from sparseml.tensorflow_v1.utils import (
    clean_tensor_name,
    get_ops_and_inputs_by_name_or_regex,
    get_tensor_var,
    is_prunable_op,
    tf_compat,
    tf_compat_div,
)


__all__ = [
    "PruningOpVars",
    "PruningScope",
    "create_op_pruning",
    "create_graph_ops_pruning",
    "create_ks_scheduled_constant_graph_ops",
    "get_or_create_graph_ops_pruning",
    "apply_op_vars_masks",
    "create_summaries_pruning",
    "create_ks_schedule_ops",
    "get_or_create_ks_schedule_ops",
    "get_or_create_ks_scheduled_graph_ops",
]


PruningOpVars = namedtuple(
    "PruningOpVars", ["op", "op_input", "update", "mask", "masked"]
)


class PruningScope(object):
    """
    Convenience class for dealing with scope and names for kernel sparsity
    in the tf graph.
    """

    NM_KS = "nm_ks"
    NM_KS_OPS = "nm_ks_ops"

    OPS = "ops"
    OPS_INPUT = "input_ops"
    OPS_UPDATE = "update_ops"
    OPS_SUMMARY = "summary_ops"
    OPS_SCHEDULE = "schedule_ops"
    OPS_SPARSITY = "sparsity_ops"

    OP_COND_UPDATE = "nm_conditional_update"
    OP_SPARSITY = "nm_sparsity"
    OP_UPDATE_READY = "nm_update_ready"
    OP_MASKED_VAR = "nm_masked_var"
    OP_MASK_ASSIGN = "nm_mask_assign"
    OP_PRUNE_VARS_ASSIGN = "nm_prune_vars_assign"
    OP_MASK_UPDATE_NO_OP = "nm_mask_update_no_op"
    OP_MASK_UPDATE = "nm_mask_update"
    OP_WEIGHT_UPDATE = "nm_weight_update"
    OP_SAVE = "nm_save"

    VAR_MASK = "nm_mask"
    VAR_THRESHOLD = "nm_threshold"

    @staticmethod
    def general(ks_group: str, additional: str = None, trailing_slash: bool = False):
        """
        Create a general kernel sparsity scope in the tf graph.
        Use cases are for generic ops like target sparsity, conditional updates, etc.

        :param ks_group: the group identifier the scope should be created under
        :param additional: any additional scope that should be added to the end
        :param trailing_slash: include a trailing forward slash if True, else False
        :return: the proper scope
        """
        scope = PruningScope._format(PruningScope.NM_KS_OPS, ks_group)
        scope = PruningScope._format(
            scope, additional=additional, trailing_slash=trailing_slash
        )

        return scope

    @staticmethod
    def model(
        op_tens: tf_compat.Tensor,
        ks_group: str,
        additional: str = None,
        trailing_slash: bool = False,
    ) -> str:
        """
        Create a model specific kernel sparsity scope in the tf graph.
        Use cases are for the specific mask, threshold, etc variables
        to induce sparsity along with the ops to update those vars.

        :param op_tens: the op tensor to create the scope for
        :param ks_group: the group identifier the scope should be created under
        :param additional: any additional scope that should be added to the end
        :param trailing_slash: include a trailing forward slash if True, else False
        :return: the proper scope
        """
        op_name = clean_tensor_name(op_tens)
        scope = PruningScope._format(
            "{}_{}".format(op_name, PruningScope.NM_KS), ks_group
        )
        scope = PruningScope._format(
            scope, additional=additional, trailing_slash=trailing_slash
        )

        return scope

    @staticmethod
    def collection_name(ks_group: str, name: str) -> str:
        """
        Create a predictable name for a given variable / op in a group for lookup /
        storage in a collection

        :param ks_group: the group identifier the name belongs under
        :param name: the name of the op or variable to be stored or retrieved
        :return: the formatted name for use in a collection
        """
        return "nm_ks_collection_{}_{}".format(ks_group, name)

    @staticmethod
    def _format(
        current: str, additional: str = None, trailing_slash: bool = False
    ) -> str:
        scope = current

        if additional is not None:
            scope = "{}/{}".format(current, additional)

        if trailing_slash:
            scope += "/"

        return scope


def create_op_pruning_no_update(
    op: tf_compat.Operation,
    op_input: tf_compat.Tensor,
    ks_group: str,
    leave_enabled: bool = True,
    is_after_end_step: tf_compat.Tensor = None,
) -> PruningOpVars:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to an operators variable without returning a
    PruningOpVars.update value.

    :param op: the operation to prune to the given sparsity
    :param op_input: the parameter within the op to create a mask for
    :param ks_group: the group identifier the scope should be created under
        mask_creator
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking
    :param is_after_end_step: only should be provided if leave_enabled is False;
        tensor that is true if the current global step is after end_epoch
    :return: a named tuple containing the assignment op, mask variable,
        threshold tensor, and masked tensor
    """
    if tf_contrib_err:
        raise tf_contrib_err

    op_sgv = graph_editor.sgv(op)

    # create the necessary variables first
    with tf_compat.variable_scope(
        PruningScope.model(op, ks_group), reuse=tf_compat.AUTO_REUSE
    ):
        mask = tf_compat.get_variable(
            PruningScope.VAR_MASK,
            op_input.get_shape(),
            initializer=tf_compat.ones_initializer(),
            trainable=False,
            dtype=op_input.dtype,
        )
    tf_compat.add_to_collection(
        PruningScope.collection_name(ks_group, PruningScope.VAR_MASK), mask
    )

    # create the masked operation and assign as the new input to the op
    with tf_compat.name_scope(PruningScope.model(op, ks_group, trailing_slash=True)):
        masked = tf_compat.multiply(mask, op_input, PruningScope.OP_MASKED_VAR)
        op_inp_tens = (
            masked
            if leave_enabled
            else tf_compat.cond(is_after_end_step, lambda: op_input, lambda: masked)
        )
        op_swapped_inputs = [
            inp if inp != op_input else op_inp_tens for inp in op_sgv.inputs
        ]
        graph_editor.swap_inputs(op, op_swapped_inputs)
    tf_compat.add_to_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_MASKED_VAR), masked
    )
    return PruningOpVars(op, op_input, None, mask, masked)


def create_op_pruning(
    op: tf_compat.Operation,
    op_input: tf_compat.Tensor,
    sparsity: tf_compat.Tensor,
    update_ready: tf_compat.Tensor,
    leave_enabled: bool,
    is_after_end_step: tf_compat.Tensor,
    ks_group: str,
    mask_creator: PruningMaskCreator,
) -> PruningOpVars:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to an operators variable.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param op: the operation to prune to the given sparsity
    :param op_input: the variable of the parameter within op to prune
    :param sparsity: the target sparsity to use for assigning the masks
    :param update_ready: the tensor where if true will update the mask from sparsity,
        if false will not update the mask
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking
    :param is_after_end_step: tensor that is true if the current global step
        is after end_epoch
    :param ks_group: the group identifier the scope should be created under
    :param mask_creator: object to define sparisty mask creation
    :return: a named tuple containing the assignment op, mask variable,
        threshold tensor, and masked tensor
    """
    initial_vars = create_op_pruning_no_update(
        op, op_input, ks_group, leave_enabled, is_after_end_step
    )
    op = initial_vars.op
    op_var_tens = initial_vars.op_input
    mask = initial_vars.mask
    masked = initial_vars.masked

    def _update():
        # create the update ops using the target sparsity tensor
        with tf_compat.name_scope(
            PruningScope.model(
                op,
                ks_group,
                additional=PruningScope.OPS_UPDATE,
                trailing_slash=True,
            )
        ):
            new_mask = mask_creator.create_sparsity_mask(op_var_tens, sparsity)
            weight_var = get_tensor_var(op_var_tens)
            return tf_compat.group(
                tf_compat.assign(mask, new_mask, name=PruningScope.OP_MASK_ASSIGN),
                tf_compat.assign(
                    weight_var,
                    tf_compat.multiply(new_mask, op_var_tens),
                    name=PruningScope.OP_WEIGHT_UPDATE,
                ),
            )

    def _no_update():
        with tf_compat.name_scope(
            PruningScope.model(
                op,
                ks_group,
                additional=PruningScope.OPS_UPDATE,
                trailing_slash=True,
            )
        ):
            # return no op wrapped in group to match update type
            return tf_compat.group(
                tf_compat.constant(
                    0.0, dtype=op_var_tens.dtype, name=PruningScope.OP_MASK_UPDATE_NO_OP
                )
            )

    with tf_compat.name_scope(
        PruningScope.model(
            op,
            ks_group,
            additional=PruningScope.OPS_UPDATE,
            trailing_slash=True,
        )
    ):
        mask_update = tf_compat.cond(
            update_ready, _update, _no_update, name=PruningScope.OP_MASK_UPDATE
        )

    # add return state to collections
    tf_compat.add_to_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_MASK_UPDATE), mask_update
    )

    return PruningOpVars(op, op_var_tens, mask_update, mask, masked)


def create_constant_op_pruning(
    op: tf_compat.Operation,
    op_input: tf_compat.Tensor,
    is_start_step: tf_compat.Tensor,
    is_end_step: tf_compat.Tensor,
    ks_group: str,
) -> PruningOpVars:
    """
    Creates PruningOpVars with constant mask for the given operation
    on start step, sets mask to be all 1s for the weight tensor where
    the operation input is non zero and 0 elsewhere.
    At the end_step we revert the mask to be all 1s and update the weight.

    :param op: the operation to prune to the given sparsity
    :param op_input: the input tensor to op to create a constant mask for
    :param is_start_step: True only if we are at the start step.
    :param is_end_step: True only if we are at the start end step.
    :param ks_group: the group identifier the scope should be created under
    :return: a named tuple containing the assignment op, mask variable,
        threshold tensor, and masked tensor
    """
    initial_vars = create_op_pruning_no_update(op, op_input, ks_group)
    op = initial_vars.op
    op_var_tens = initial_vars.op_input
    mask = initial_vars.mask
    masked = initial_vars.masked

    is_start_or_end_step = tf_compat.logical_or(is_start_step, is_end_step)

    def _set_constant_mask():
        # Assign mask tensor to be 1 for all nonzero values of op_var_tens otherwise 0
        # On end step, revert mask to be all 1s
        with tf_compat.name_scope(
            PruningScope.model(
                op,
                ks_group,
                additional=PruningScope.OPS_UPDATE,
                trailing_slash=True,
            )
        ):
            new_mask = tf_compat.cond(
                is_start_step,
                lambda: tf_compat.cast(
                    tf_compat.not_equal(op_var_tens, 0.0), dtype=op_var_tens.dtype
                ),
                lambda: tf_compat.ones(op_var_tens.shape, dtype=op_var_tens.dtype),
            )
            weight_var = get_tensor_var(op_var_tens)
            return tf_compat.group(
                tf_compat.assign(mask, new_mask, name=PruningScope.OP_MASK_ASSIGN),
                tf_compat.assign(
                    weight_var, masked, name=PruningScope.OP_WEIGHT_UPDATE
                ),
            )

    def _no_op():
        with tf_compat.name_scope(
            PruningScope.model(
                op,
                ks_group,
                additional=PruningScope.OPS_UPDATE,
                trailing_slash=True,
            )
        ):
            # return no op wrapped in group to match update type
            return tf_compat.group(
                tf_compat.constant(
                    0.0, dtype=op_var_tens.dtype, name=PruningScope.OP_MASK_UPDATE_NO_OP
                )
            )

    with tf_compat.name_scope(
        PruningScope.model(
            op,
            ks_group,
            additional=PruningScope.OPS_UPDATE,
            trailing_slash=True,
        )
    ):
        mask_update = tf_compat.cond(
            is_start_or_end_step,
            _set_constant_mask,
            _no_op,
            name=PruningScope.OP_MASK_UPDATE,
        )
    return PruningOpVars(op, op_var_tens, mask_update, mask, masked)


def create_graph_ops_pruning(
    graph: tf_compat.Graph,
    var_names: List[str],
    sparsity: tf_compat.Tensor,
    update_ready: tf_compat.Tensor,
    leave_enabled: bool,
    is_after_end_step: tf_compat.Tensor,
    ks_group: str,
    mask_creator: PruningMaskCreator,
) -> List[PruningOpVars]:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to a given list of operators in a graph.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param var_names: the names or regex patterns of names of variables to prune in the
        graph to the given sparsity
    :param sparsity: the target sparsity to use for assigning the masks
    :param update_ready: the tensor where if true will update the mask from sparsity,
        if false will not update the mask
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking
    :param is_after_end_step: tensor that is true if the current global step
        is after end_epoch
    :param ks_group: the group identifier the scope should be created under
    :param mask_creator: optional object to define sparisty mask creation
    :return: a list of the created named tuples each containing the
        assignment op, mask variable, threshold tensor, and masked tensor
    """
    pruning_op_vars = []
    variable_masks = {}  # cache of mask vars for input variables

    for op, op_input in get_ops_and_inputs_by_name_or_regex(var_names, graph):
        if op_input not in variable_masks:
            op_vars = create_op_pruning(
                op,
                op_input,
                sparsity,
                update_ready,
                leave_enabled,
                is_after_end_step,
                ks_group,
                mask_creator,
            )
            pruning_op_vars.append(op_vars)
            variable_masks[op_input] = op_vars
        else:  # Reuse masks if the input variable is shared and already computed
            _, _, mask_update, mask, masked = variable_masks[op_input]
            pruning_op_vars.append(
                PruningOpVars(op, op_input, mask_update, mask, masked)
            )
        tf_compat.add_to_collection(
            PruningScope.collection_name(ks_group, PruningScope.OPS), op
        )
        tf_compat.add_to_collection(
            PruningScope.collection_name(ks_group, PruningScope.OPS_INPUT), op_input
        )
    return pruning_op_vars


def get_or_create_graph_ops_pruning(
    graph: tf_compat.Graph,
    var_names: List[str],
    sparsity: tf_compat.Tensor,
    update_ready: tf_compat.Tensor,
    leave_enabled: bool,
    is_after_end_step: tf_compat.Tensor,
    ks_group: str,
    mask_creator: PruningMaskCreator,
) -> List[PruningOpVars]:
    """
    Creates or retrieves (if previously created) the necessary variables
    and operators to gradually apply sparsity to a given list of operators in a graph.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param var_names: the names or regex patterns of names of variables to prune in the
        graph to the given sparsity
    :param sparsity: the target sparsity to use for assigning the masks
    :param update_ready: the tensor where if true will update the mask from sparsity,
        if false will not update the mask
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking
    :param is_after_end_step: tensor that is true if the current global step
        is after end_epoch
    :param ks_group: the group identifier the scope should be created under
    :param mask_creator: optional object to define sparisty mask creation
    :return: a list of the created or retrieved named tuples each containing the
        assignment op, mask variable, threshold tensor, and masked tensor
    """
    ops = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OPS)
    )
    ops_input = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OPS_INPUT)
    )
    mask_updates = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_MASK_UPDATE)
    )
    masks = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.VAR_MASK)
    )
    maskeds = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_MASKED_VAR)
    )

    if (
        len(ops) < 1
        or len(ops_input) < 1
        or len(mask_updates) < 1
        or len(masks) < 1
        or len(maskeds) < 1
    ):  # create new pruning ops
        pruning_op_vars = create_graph_ops_pruning(
            graph,
            var_names,
            sparsity,
            update_ready,
            leave_enabled,
            is_after_end_step,
            ks_group,
            mask_creator,
        )
    else:  # use collection pruning ops
        pruning_op_vars = []
        for op, op_input, mask_update, mask, masked in zip(
            ops, ops_input, mask_updates, masks, maskeds
        ):
            pruning_op_vars.append(
                PruningOpVars(op, op_input, mask_update, mask, masked)
            )

    return pruning_op_vars


def create_summaries_pruning(pruning_op_vars: List[PruningOpVars]):
    """
    Create TensorBoard summary ops in the current graph for the
    given list of PruningOpVars.

    :param pruning_op_vars: the list of named tuples containing the masked input to the
        pruned op to record sparsity for in TensorBoard.
    :return: the created summaries for the pruned op vars
    """
    summaries = []

    for op_vars in pruning_op_vars:
        try:
            zero_fraction = tf_compat.zero_fraction
        except Exception:

            def zero_fraction(inp: tf_compat.Tensor):
                nonzero = tf_compat.cast(
                    tf_compat.reduce_sum(
                        tf_compat.cast(tf_compat.not_equal(inp, 0), tf_compat.int64)
                    ),
                    tf_compat.float32,
                )
                size = tf_compat.size(inp, out_type=tf_compat.float32)

                return 1 - tf_compat_div(nonzero, size)

        if is_prunable_op(op_vars.op):
            sum_op = tf_compat.summary.scalar(
                "Modifier_Pruning/{}".format(clean_tensor_name(op_vars.op)),
                zero_fraction(op_vars.masked),
            )
            summaries.append(sum_op)

    return summaries


def apply_op_vars_masks(
    pruning_op_vars: List[PruningOpVars], ks_group: str, sess: tf_compat.Session
):
    """
    Apply the masks to the original ops input var so that it can be saved
    with the desired sparsity for later.

    :param pruning_op_vars: the list of named tuples containing the sparse mask
        and the op variable to apply the sparse mask to
    :param ks_group: the group to create the assign ops under
    :param sess: the session to use to run the assign
    """
    for op_vars in pruning_op_vars:
        with tf_compat.name_scope(
            PruningScope.model(op_vars.op, ks_group, PruningScope.OP_SAVE)
        ):
            masked_var = tf_compat.multiply(op_vars.op_input, op_vars.mask)
            input_var = get_tensor_var(op_vars.op_input)
            assign = tf_compat.assign(input_var, masked_var)
            sess.run(assign)


def create_ks_schedule_ops(
    global_step: tf_compat.Variable,
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    Create a gradual schedule for model pruning (kernel sparsity).
    Creates a sparsity tensor that goes from init_sparsity til final_sparsity
    starting at begin_step and ending at end_step.
    Uses the global_step to map those.
    Additionally creates an update_ready tensor that is True if an update
    to the sparsity tensor should be run, False otherwise.

    :param global_step: the global optimizer step for the training graph
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a
        weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between
        init_sparsity and final_sparsity higher values will lead to larger sparsity
        steps at the beginning vs the end ie: linear (1) vs cubic (3)
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the signal for update_ready and the target sparsity
    """

    # create the scheduling ops first and the sparsity ops
    with tf_compat.name_scope(
        PruningScope.general(
            ks_group, additional=PruningScope.OPS_SCHEDULE, trailing_slash=True
        )
    ):
        sched_before = tf_compat.less(global_step, begin_step)
        sched_start = tf_compat.equal(global_step, begin_step)
        sched_end = tf_compat.equal(global_step, end_step)
        sched_active = tf_compat.logical_and(
            tf_compat.greater(global_step, begin_step),
            tf_compat.less(global_step, end_step),
        )
        sched_active_inclusive = tf_compat.logical_or(
            sched_active, tf_compat.logical_or(sched_start, sched_end)
        )
        sched_update = tf_compat.cond(
            tf_compat.less_equal(update_step_freq, 0),
            lambda: tf_compat.constant(True),
            lambda: tf_compat.equal(
                tf_compat.mod((global_step - begin_step), update_step_freq), 0
            ),
        )
        sched_update_ready = tf_compat.logical_or(
            tf_compat.logical_or(sched_start, sched_end), sched_update
        )

        percentage = tf_compat.minimum(
            1.0,
            tf_compat.maximum(
                0.0,
                tf_compat_div(
                    tf_compat.cast(global_step - begin_step, tf_compat.float32),
                    end_step - begin_step,
                ),
            ),
        )
        exp_percentage = 1 - tf_compat.pow(1 - percentage, exponent)
        calc_sparsity = (
            tf_compat.multiply(final_sparsity - init_sparsity, exp_percentage)
            + init_sparsity
        )

        # create the update ready tensor and sparsity tensor
    with tf_compat.name_scope(PruningScope.general(ks_group, trailing_slash=True)):
        update_ready = tf_compat.logical_and(
            sched_active_inclusive,
            sched_update_ready,
            name=PruningScope.OP_UPDATE_READY,
        )
        sparsity = tf_compat.case(
            [
                (sched_before, lambda: tf_compat.constant(0.0)),
                (sched_start, lambda: tf_compat.constant(init_sparsity)),
                (sched_active, lambda: calc_sparsity),
            ],
            default=lambda: tf_compat.constant(final_sparsity),
            name=PruningScope.OP_SPARSITY,
        )

        # add return state to collections
    tf_compat.add_to_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_UPDATE_READY),
        update_ready,
    )
    tf_compat.add_to_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_SPARSITY), sparsity
    )

    return update_ready, sparsity


def get_or_create_ks_schedule_ops(
    global_step: tf_compat.Tensor,
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    Creates or retrieves (if previously created) a gradual schedule
    for model pruning (kernel sparsity).
    Creates a sparsity tensor that goes from init_sparsity til final_sparsity
    starting at begin_step and ending at end_step.
    Uses the global_step to map those.
    Additionally creates an update_ready tensor that is True if an update
    to the sparsity tensor should be run, False otherwise.

    :param global_step: the global optimizer step for the training graph
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a
        weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between
        init_sparsity and final_sparsity higher values will lead to larger sparsity
        steps at the beginning vs the end ie: linear (1) vs cubic (3)
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the signal for update_ready and the target sparsity
    """
    update_ready = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_UPDATE_READY)
    )
    sparsity = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_SPARSITY)
    )

    update_ready = update_ready[0] if len(update_ready) > 0 else None
    sparsity = sparsity[0] if len(sparsity) > 0 else None

    if update_ready is None or sparsity is None:
        update_ready, sparsity = create_ks_schedule_ops(
            global_step,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            ks_group,
        )
        # add return state to collections
        tf_compat.add_to_collection(
            PruningScope.collection_name(ks_group, PruningScope.OP_UPDATE_READY),
            update_ready,
        )
        tf_compat.add_to_collection(
            PruningScope.collection_name(ks_group, PruningScope.OP_SPARSITY), sparsity
        )

    return update_ready, sparsity


def get_scheduled_update_op(
    pruning_op_vars: List[PruningOpVars],
    ks_group: str,
):
    """
    Creates model pruning (kernel sparsity) ops and vars in the graph
    to be applied over a specific schedule.
    Creates them for the ops in the graph such that they follow the given schedule.

    :param pruning_op_vars: List of tuples of operation tensors and masks.
    :param ks_group: the group identifier the scope should be created under
    :return: the update operation to run in a session
    """
    update_op = tf_compat.get_collection(
        PruningScope.collection_name(ks_group, PruningScope.OP_COND_UPDATE)
    )
    update_op = update_op[0] if len(update_op) > 0 else None

    if update_op is None:
        update_op = tf_compat.group(*[op_var.update for op_var in pruning_op_vars])

        # add return state to collections
        tf_compat.add_to_collection(
            PruningScope.collection_name(ks_group, PruningScope.OP_COND_UPDATE),
            update_op,
        )

    return update_op


def get_or_create_ks_scheduled_graph_ops(
    graph: tf_compat.Graph,
    global_step: tf_compat.Variable,
    var_names: List[str],
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    leave_enabled: bool,
    ks_group: str,
    mask_creator: PruningMaskCreator,
) -> Tuple[tf_compat.Tensor, List[PruningOpVars], tf_compat.Tensor, tf_compat.Tensor]:
    """
    Gets or creates model pruning (kernel sparsity) ops and vars in the graph
    to be applied over a specific schedule.
    Creates them for the var_names in the graph such that they follow a schedule
    from begin_step to end_step starting at init_sparsity and ending at final_sparsity.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param global_step: the global optimizer step for the training graph
    :param var_names: the names or regex patterns of names of variables to prune in the
        graph
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a
        weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between
        init_sparsity and final_sparsity higher values will lead to larger sparsity
        steps at the beginning vs the end ie: linear (1) vs cubic (3)
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking
    :param ks_group: the group identifier the scope should be created under
    :param mask_creator: optional object to define sparisty mask creation
    :return: a tuple containing the update operation to run in a session,
        a list of the pruning ops and vars for each desired op in the graph,
        the tensor containing the update_ready signal for the pruning ops,
        the tensor containing the set sparsity for the pruning ops
    """
    update_ready, sparsity = get_or_create_ks_schedule_ops(
        global_step,
        begin_step,
        end_step,
        update_step_freq,
        init_sparsity,
        final_sparsity,
        exponent,
        ks_group,
    )
    is_after_end_step = tf_compat.greater(global_step, end_step)
    pruning_op_vars = get_or_create_graph_ops_pruning(
        graph,
        var_names,
        sparsity,
        update_ready,
        leave_enabled,
        is_after_end_step,
        ks_group,
        mask_creator,
    )
    update_op = get_scheduled_update_op(pruning_op_vars, ks_group)
    return update_op, pruning_op_vars, update_ready, sparsity


def create_ks_scheduled_constant_graph_ops(
    graph: tf_compat.Graph,
    global_step: tf_compat.Variable,
    var_names: List[str],
    begin_step: int,
    end_step: int,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, List[PruningOpVars]]:
    """
    Creates constant model pruning ops.  Does not modify the graph.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param global_step: the global optimizer step for the training graph
    :param var_names: a list of names or regex patterns to create constant ops
        for within the graph
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the update operation to run in a session,
        a list of the pruning ops and vars for each desired op in the graph
    """
    pruning_op_vars = []
    is_start_step = tf_compat.equal(global_step, begin_step)
    is_end_step = tf_compat.equal(global_step, end_step)
    for op, op_input in get_ops_and_inputs_by_name_or_regex(var_names, graph):
        op_vars = create_constant_op_pruning(
            op, op_input, is_start_step, is_end_step, ks_group
        )
        pruning_op_vars.append(op_vars)

    update_op = get_scheduled_update_op(pruning_op_vars, ks_group)
    return update_op, pruning_op_vars
