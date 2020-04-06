"""
Code related to applying a mask onto a variable to impose kernel sparsity,
aka model pruning, on a TensorFlow graph.
"""

from typing import Tuple, List, Union
from collections import namedtuple
import tensorflow.contrib.graph_editor as ge

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    tf_compat_div,
    get_var_name,
    get_op_input_var,
)


__all__ = [
    "PruningOpVars",
    "KSScope",
    "create_op_pruning",
    "create_graph_ops_pruning",
    "get_or_create_graph_ops_pruning",
    "create_ks_schedule_ops",
    "get_or_create_ks_schedule_ops",
    "get_or_create_ks_scheduled_graph_ops",
]


PruningOpVars = namedtuple("PruningOpVars", ["assign", "mask", "thresh", "masked"])


class KSScope(object):
    """
    Convenience class for dealing with scope and names for kernel sparsity
    in the tf graph.
    """

    NM_KS = "nm_ks"
    NM_KS_OPS = "nm_ks_ops"

    OPS_UPDATE = "update_ops"
    OPS_SCHEDULE = "schedule_ops"
    OPS_SPARSITY = "sparsity_ops"

    OP_COND_UPDATE = "nm_conditional_update"
    OP_SPARSITY = "nm_sparsity"
    OP_UPDATE_READY = "nm_update_ready"
    OP_MASKED_VAR = "nm_masked_var"
    OP_MASK_ASSIGN = "nm_mask_assign"
    OP_THRESH_ASSIGN = "nm_threshold_assign"
    OP_PRUNE_VARS_ASSIGN = "nm_prune_vars_assign"

    NO_OP_MASK_UPDATE = "nm_mask_update"
    NO_OP_MASK_NO_UPDATE = "nm_mask_no_update"

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
        scope = KSScope._format(KSScope.NM_KS_OPS, ks_group)
        scope = KSScope._format(
            scope, additional=additional, trailing_slash=trailing_slash
        )

        return scope

    @staticmethod
    def model(
        var_tens: tf_compat.Tensor,
        ks_group: str,
        additional: str = None,
        trailing_slash: bool = False,
    ) -> str:
        """
        Create a model specific kernel sparsity scope in the tf graph.
        Use cases are for the specific mask, threshold, etc variables
        to induce sparsity along with the ops to update those vars.

        :param var_tens: the variable tensor to create the scope for
        :param ks_group: the group identifier the scope should be created under
        :param additional: any additional scope that should be added to the end
        :param trailing_slash: include a trailing forward slash if True, else False
        :return: the proper scope
        """
        var_name = get_var_name(var_tens)
        scope = KSScope._format("{}_{}".format(var_name, KSScope.NM_KS), ks_group)
        scope = KSScope._format(
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


def create_op_pruning(
    op: tf_compat.Operation,
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    ks_group: str,
) -> PruningOpVars:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to an operators variable.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param op: the operation to prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param ks_group: the group identifier the scope should be created under
    :return: a named tuple containing the assignment op, mask variable,
        threshold tensor, and masked tensor
    """
    op_sgv = ge.sgv(op)
    op_var_tens = get_op_input_var(op, var_index)

    # create the necessary variables first
    with tf_compat.variable_scope(
        KSScope.model(op_var_tens, ks_group), reuse=tf_compat.AUTO_REUSE
    ):
        mask = tf_compat.get_variable(
            KSScope.VAR_MASK,
            op_var_tens.get_shape(),
            initializer=tf_compat.ones_initializer(),
            trainable=False,
            dtype=op_var_tens.dtype,
        )

    # create the masked operation and assign as the new input to the op
    with tf_compat.name_scope(
        KSScope.model(op_var_tens, ks_group, trailing_slash=True)
    ):
        masked = tf_compat.math.multiply(mask, op_var_tens, KSScope.OP_MASKED_VAR)
        op_swapped_inputs = [
            inp if inp != op_var_tens else masked for inp in op_sgv.inputs
        ]
        ge.swap_inputs(op, op_swapped_inputs)

    # create the update ops using the target sparsity tensor
    with tf_compat.name_scope(
        KSScope.model(
            op_var_tens, ks_group, additional=KSScope.OPS_UPDATE, trailing_slash=True
        )
    ):
        abs_var = tf_compat.abs(op_var_tens)
        sparse_index = tf_compat.cast(
            tf_compat.math.round(
                tf_compat.cast(tf_compat.size(abs_var), tf_compat.dtypes.float32)
                * (1.0 - sparsity)
            ),
            tf_compat.dtypes.int32,
        )
        sparse_index = tf_compat.maximum(sparse_index - 1, 0)
        sorted_vals, _ = tf_compat.math.top_k(
            tf_compat.reshape(abs_var, [-1]), k=tf_compat.size(abs_var)
        )
        threshold = tf_compat.gather(
            sorted_vals, sparse_index, name=KSScope.VAR_THRESHOLD
        )
        new_mask = tf_compat.cast(
            tf_compat.greater(abs_var, threshold), tf_compat.dtypes.float32
        )
        mask_assign = tf_compat.assign(mask, new_mask, name=KSScope.OP_MASK_ASSIGN)

    # add return state to collections
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_ASSIGN), mask_assign
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_MASK), mask
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_THRESHOLD), threshold
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASKED_VAR), masked
    )

    return PruningOpVars(mask_assign, mask, threshold, masked)


def create_graph_ops_pruning(
    graph: tf_compat.Graph,
    op_names: List[str],
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    ks_group: str,
) -> List[PruningOpVars]:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to a given list of operators in a graph.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param op_names: the list of name of the operations in the
        graph to prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param ks_group: the group identifier the scope should be created under
    :return: a list of the created named tuples each containing the
        assignment op, mask variable, threshold tensor, and masked tensor
    """
    pruning_op_vars = []

    for op_name in op_names:
        op = graph.get_operation_by_name(op_name)
        op_vars = create_op_pruning(op, var_index, sparsity, ks_group)
        pruning_op_vars.append(op_vars)

    return pruning_op_vars


def get_or_create_graph_ops_pruning(
    graph: tf_compat.Graph,
    op_names: List[str],
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    ks_group: str,
) -> List[PruningOpVars]:
    """
    Creates or retrieves (if previously created) the necessary variables
    and operators to gradually apply sparsity to a given list of operators in a graph.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param op_names: the list of name of the operations in the graph to
        prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param ks_group: the group identifier the scope should be created under
    :return: a list of the created or retrieved named tuples each containing the
        assignment op, mask variable, threshold tensor, and masked tensor
    """
    mask_assigns = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_ASSIGN)
    )
    masks = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_MASK)
    )
    thresholds = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_THRESHOLD)
    )
    maskeds = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASKED_VAR)
    )

    if (
        len(mask_assigns) < 1
        or len(masks) < 1
        or len(thresholds) < 1
        or len(maskeds) < 1
    ):
        pruning_op_vars = create_graph_ops_pruning(
            graph, op_names, var_index, sparsity, ks_group
        )
    else:
        pruning_op_vars = []

        for mask_assign, mask, threshold, masked in zip(
            mask_assigns, masks, thresholds, maskeds
        ):
            pruning_op_vars.append(PruningOpVars(mask_assign, mask, threshold, masked))

    return pruning_op_vars


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

    # create the scheduling ops first
    with tf_compat.name_scope(
        KSScope.general(ks_group, additional=KSScope.OPS_SCHEDULE, trailing_slash=True)
    ):
        sched_active = tf_compat.logical_and(
            tf_compat.greater_equal(global_step, begin_step),
            tf_compat.less_equal(global_step, end_step),
        )
        sched_start_end = tf_compat.logical_or(
            tf_compat.equal(global_step, begin_step),
            tf_compat.equal(global_step, end_step),
        )
        sched_update = tf_compat.logical_or(
            tf_compat.equal(
                tf_compat.mod((global_step - begin_step), update_step_freq), 0
            ),
            tf_compat.less_equal(update_step_freq, 0),
        )

    # create the update ready tensor
    with tf_compat.name_scope(KSScope.general(ks_group, trailing_slash=True)):
        update_ready = tf_compat.logical_and(
            sched_active,
            tf_compat.logical_or(sched_start_end, sched_update),
            name=KSScope.OP_UPDATE_READY,
        )

    # create the sparsity ops
    with tf_compat.name_scope(
        KSScope.general(ks_group, additional=KSScope.OPS_SPARSITY, trailing_slash=True)
    ):
        percentage = tf_compat_div(
            tf_compat.cast(global_step - begin_step, tf_compat.dtypes.float32),
            end_step - begin_step,
        )
        percentage = tf_compat.minimum(1.0, percentage)
        percentage = tf_compat.maximum(0.0, percentage)
        exp = tf_compat.pow(percentage, 1.0 / exponent)
        sparsity = tf_compat.multiply(final_sparsity - init_sparsity, exp)

    # create the sparsity tensor
    with tf_compat.name_scope(KSScope.general(ks_group, trailing_slash=True)):
        sparsity = tf_compat.add(sparsity, init_sparsity, name=KSScope.OP_SPARSITY)

    # add return state to collections
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_UPDATE_READY), update_ready
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_SPARSITY), sparsity
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
        KSScope.collection_name(ks_group, KSScope.OP_UPDATE_READY)
    )
    sparsity = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_SPARSITY)
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

    return update_ready, sparsity


def get_or_create_ks_scheduled_graph_ops(
    graph: tf_compat.Graph,
    global_step: tf_compat.Variable,
    op_names: List[str],
    var_index: Union[int, str],
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, List[PruningOpVars]]:
    """
    Gets or creates model pruning (kernel sparsity) ops and vars in the graph
    to be applied over a specific schedule.
    Creates them for the op_names in the graph such that they follow a schedule
    from begin_step to end_step starting at init_sparsity and ending at final_sparsity.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param global_step: the global optimizer step for the training graph
    :param op_names: the list of name of the operations in the graph to
        prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
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
    :return: a tuple containing the update operation to run in a session,
        and a list of the pruning ops and vars for each desired op in the graph
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

    update_op = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_COND_UPDATE)
    )
    update_op = update_op[0] if len(update_op) > 0 else None

    if update_op is not None:
        pruning_op_vars = get_or_create_graph_ops_pruning(
            graph, op_names, var_index, sparsity, ks_group
        )
    else:
        pruning_op_vars = []

        def _update_ops_wrapper():
            pruning_op_vars.extend(
                get_or_create_graph_ops_pruning(
                    graph, op_names, var_index, sparsity, ks_group
                )
            )

            with tf_compat.control_dependencies(
                [op_var.assign for op_var in pruning_op_vars]
            ):
                return tf_compat.no_op(KSScope.NO_OP_MASK_UPDATE)

        def _no_update_ops_wrapper():
            return tf_compat.no_op(KSScope.NO_OP_MASK_NO_UPDATE)

        with tf_compat.name_scope(KSScope.general(ks_group, trailing_slash=True)):
            update_op = tf_compat.cond(
                update_ready,
                _update_ops_wrapper,
                _no_update_ops_wrapper,
                name=KSScope.OP_COND_UPDATE,
            )

        # add return state to collections
        tf_compat.add_to_collection(
            KSScope.collection_name(ks_group, KSScope.OP_COND_UPDATE), update_op
        )

    return update_op, pruning_op_vars
