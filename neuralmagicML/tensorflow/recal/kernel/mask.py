"""
Code related to applying a mask onto a variable to impose kernel sparsity, aka model pruning.

Flow is made up of three steps to properly add the sparsity ops to a graph:
    1. :py:func:`~create_ks_schedule_ops` or :py:func:`~get_or_create_ks_schedule_ops`
       creates the sparsity schedule as well as the update checks
    2. :py:func:`~create_op_pruning`, :py:func:`~create_graph_ops_pruning`,
       or :py:func:`~get_or_create_graph_ops_pruning`
       creates the mask and update ops for a specific operation in the graph
    3. :py:func:`~create_ks_update_op` or :py:func:`~get_or_create_ks_update_op`
       creates the containing update op that should be run in a session after each batch
"""

from typing import Tuple, List, Union
import tensorflow.contrib.graph_editor as ge

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    tf_compat_div,
    get_var_name,
    get_op_input_var,
)


__all__ = [
    "KSScope",
    "create_ks_schedule_ops",
    "get_or_create_ks_schedule_ops",
    "create_op_pruning",
    "create_graph_ops_pruning",
    "get_or_create_graph_ops_pruning",
    "create_ks_update_op",
    "get_or_create_ks_update_op",
]


class KSScope(object):
    """
    Convenience class for dealing with scope and names for kernel sparsity in the tf graph
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

    NO_OP_MASK_UPDATE = "nm_mask_update"
    NO_OP_MASK_NO_UPDATE = "nm_mask_no_update"

    VAR_MASK = "nm_mask"
    VAR_THRESHOLD = "nm_threshold"

    @staticmethod
    def general(ks_group: str, additional: str = None, trailing_slash: bool = False):
        """
        Create a general kernel sparsity scope in the tf graph.
        Use cases are for generic ops like target sparsity, conditional updates, etc

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
        Use cases are for the specific mask, threshold, etc variables to induce sparsity.
        Along with the ops to update those vars

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
        Create a predictable name for a given variable / op in a group for lookup / storage in a collection

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


def create_ks_schedule_ops(
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    global_step: tf_compat.Variable,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    First step for adding masks for a group to a tf graph.
    Creates the graph for calculating when an update should happen
    as well as what the target sparsity should be at the current global step.

    update_ready should be used with :py:func:`~create_ks_update_ops`
    sparsity should be used with :py:func:`~create_op_pruning`

    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between init_sparsity and final_sparsity
                     higher values will lead to larger sparsity steps at the beginning vs the end
                     ie: linear (1) vs cubic (3)
    :param global_step: the global optimizer step for the training graph
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
        sched_update = tf_compat.logical_or(
            tf_compat.equal(
                tf_compat.mod((global_step - begin_step), update_step_freq), 0
            ),
            tf_compat.less_equal(update_step_freq, 0),
        )

    # create the update ready tensor
    with tf_compat.name_scope(KSScope.general(ks_group, trailing_slash=True)):
        update_ready = tf_compat.logical_and(
            sched_active, sched_update, name=KSScope.OP_UPDATE_READY
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
        exp = tf_compat.pow(1.0 - percentage, exponent)
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
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    global_step: tf_compat.Variable,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    gets previous ops if available (ie recalled on the same graph for the same group again)
    if none, then creates the op

    see :py:func:`~create_ks_schedule_ops` for more details

    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between init_sparsity and final_sparsity
                     higher values will lead to larger sparsity steps at the beginning vs the end
                     ie: linear (1) vs cubic (3)
    :param global_step: the global optimizer step for the training graph
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
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            global_step,
            ks_group,
        )

    return update_ready, sparsity


def create_op_pruning(
    op: tf_compat.Operation,
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    Middle step for adding masks for a operator's group in the tf graph.
    Must be called once for each desired op to be added to the graph.
    Creates the necessary variables and operators to gradually
    apply sparsity to an operators variable.

    sparsity should be taken from :py:func:`~create_ks_schedule_ops`
    the return mask_assign and thresh_assign should be inputs to :py:func:`~create_ks_update_ops`

    :param op: the operation to prune to the given sparsity
    :param var_index: the index for where the variable is, see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the mask assignment op and the threshold assignment op
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
        threshold = tf_compat.get_variable(
            KSScope.VAR_THRESHOLD,
            [],
            initializer=tf_compat.zeros_initializer(),
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
        sparse_index = (
            tf_compat.cast(
                tf_compat.math.round(
                    tf_compat.cast(tf_compat.size(abs_var), tf_compat.dtypes.float32)
                    * (1.0 - sparsity)
                ),
                tf_compat.dtypes.int32,
            )
            - 1
        )
        sorted_vals, _ = tf_compat.math.top_k(
            tf_compat.reshape(abs_var, [-1]), k=tf_compat.size(abs_var)
        )
        new_threshold = tf_compat.gather(sorted_vals, sparse_index)
        new_mask = tf_compat.cast(
            tf_compat.greater(abs_var, threshold), tf_compat.dtypes.float32
        )

        mask_assign = tf_compat.assign(mask, new_mask, name=KSScope.OP_MASK_ASSIGN)
        thresh_assign = tf_compat.assign(
            threshold, new_threshold, name=KSScope.OP_THRESH_ASSIGN
        )

    # add return state to collections
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_MASK), mask
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_THRESHOLD), threshold
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASKED_VAR), masked
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_ASSIGN), mask_assign
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_THRESH_ASSIGN), thresh_assign
    )

    return mask_assign, thresh_assign


def create_graph_ops_pruning(
    graph: tf_compat.Graph,
    op_names: List[str],
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    ks_group: str,
) -> Tuple[List[tf_compat.Tensor], List[tf_compat.Tensor]]:
    """
    Grab the given operators by name from the graph and then apply pruning to them.
    Returns the assigment ops for pruning the given operators.

    See :py:func:`~create_op_pruning` for more details

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param op_names: the list of name of the operations in the graph to prune to the given sparsity
    :param var_index: the index for where the variable is, see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the mask assignment op and the threshold assignment op
    """
    mask_assign_ops = []
    thresh_assign_ops = []

    for op_name in op_names:
        op = graph.get_operation_by_name(op_name)
        mask_assign, threshold_assign = create_op_pruning(
            op, var_index, sparsity, ks_group
        )
        mask_assign_ops.append(mask_assign)
        thresh_assign_ops.append(threshold_assign)

    return mask_assign_ops, thresh_assign_ops


def get_or_create_graph_ops_pruning(
    graph: tf_compat.Graph,
    op_names: List[str],
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    ks_group: str,
) -> Tuple[List[tf_compat.Tensor], List[tf_compat.Tensor]]:
    """
    gets previous ops if available (ie recalled on the same graph for the same group again)
    if none, then creates the op

    see :py:func:`~create_graph_ops_pruning` for more details

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param op_names: the list of name of the operations in the graph to prune to the given sparsity
    :param var_index: the index for where the variable is, see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the mask assignment op and the threshold assignment op
    """
    mask_assign_ops = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_ASSIGN)
    )
    thresh_assign_ops = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_THRESH_ASSIGN)
    )

    if len(mask_assign_ops) < 1 or len(thresh_assign_ops) < 1:
        mask_assign_ops, thresh_assign_ops = create_graph_ops_pruning(
            graph, op_names, var_index, sparsity, ks_group
        )

    return mask_assign_ops, thresh_assign_ops


def create_ks_update_op(
    update_ready: tf_compat.Tensor, assign_ops: List[tf_compat.Tensor], ks_group: str
) -> tf_compat.Tensor:
    """
    Final step for adding masks for a group to a tf graph.
    Creates the conditional update op that must be run in a session.
    If an update is ready (based on the update_ready), will run all of the assign ops.
    Otherwise will run a no_op for no update.

    update_ready tensor should be pulled from :py:func:`~create_ks_schedule_ops`
    assign_ops tensors should be pulled from :py:func:`~create_op_pruning`

    :param update_ready: the tensor that checks whether an update should happen or not
                         see :py:func:`~create_ks_schedule_ops`
    :param assign_ops: the tensors to run to update the mask and threshold assignments for applying ks
                       see :py:func:`~create_op_pruning`
    :param ks_group: the group identifier the scope should be created under
    :return: the op to be run in a session after each batch
    """

    def _update_ops_wrapper():
        with tf_compat.control_dependencies(assign_ops):
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

    return update_op


def get_or_create_ks_update_op(
    update_ready: tf_compat.Tensor, assign_ops: List[tf_compat.Tensor], ks_group: str
) -> tf_compat.Tensor:
    """
    gets previous ops if available (ie recalled on the same graph for the same group again)
    if none, then creates the op

    see :py:func:`~create_ks_update_op` for more details

    :param update_ready: the tensor that checks whether an update should happen or not
                         see :py:func:`~create_ks_schedule_ops`
    :param assign_ops: the tensors to run to update the mask and threshold assignments for applying ks
                       see :py:func:`~create_op_pruning`
    :param ks_group: the group identifier the scope should be created under
    :return: the op to be run in a session after each batch
    """
    update_op = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_COND_UPDATE)
    )
    update_op = update_op[0] if len(update_op) > 0 else None

    if update_op is None:
        update_op = create_ks_update_op(update_ready, assign_ops, ks_group)

    return update_op
