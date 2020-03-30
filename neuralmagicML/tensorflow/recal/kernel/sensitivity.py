"""
code related to calculating kernel sparsity sensitivity analysis for models
"""
from enum import Enum
import numpy
from typing import Dict, List, Union, Optional, Callable

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

from neuralmagicML.tensorflow.recal.kernel import KSScope

from neuralmagicML.recal import (
    KSLossSensitivityProgress,
    KSLossSensitivityResult,
    KSLossSensitivityAnalysis,
)

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    get_op_input_var,
)

__all__ = ["one_shot_ks_loss_sensitivity"]


_AnalysisStage = Enum(
    "_AnalysisStage", "START_OPERATION START_SPARSITY_LEVEL END_BATCH"
)


def one_shot_ks_loss_sensitivity(
    session: tf_compat.Session,
    graph: tf_compat.Graph,
    X: numpy.array,
    y: numpy.array,
    X_placeholder: tf_compat.placeholder,
    y_placeholder: tf_compat.placeholder,
    total_loss: Optional[tf_compat.Tensor],
    batch_size: int,
    samples_per_measurement: int,
    sparsity_levels: List[int] = None,
    progress_hook: Optional[Callable] = None,
) -> KSLossSensitivityAnalysis:
    """
    Run a one shot sensitivity analysis for kernel sparsity
    It does not retrain, and instead puts the model to eval mode.
    Moves layer by layer to calculate the sensitivity analysis for each and resets the previously run layers

    :param session: the current session
    :param graph: the graph of the current model
    :param X: input features, shape (n_samples, n_features)
    :param y: targets, shape (n_samples,)
    :param X_placeholder: placeholder receiving input data
    :param y_placeholder: placeholder receiving targets
    :total_loss: the loss tensor to be analyzed; if None then the default loss is used
    :param batch_size: the batch size to run through the model in eval mode
    :param samples_per_measurement: the number of samples or items to take for each measurement at each sparsity lev
    :param sparsity_levels: the sparsity levels to check for each layer to calculate sensitivity
    :prgress_hook: hook to process the progress object
    :return: the sensitivity results for every operation that is prunable
    """

    if batch_size > samples_per_measurement:
        raise ValueError("Batch size must not be greater than samples per measurement")

    if sparsity_levels is None:
        sparsity_levels = [0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]

    if progress_hook is None:
        progress_hook = KSLossSensitivityProgress.standard_update_hook()

    ops = {}
    ops.update({op.name: op for op in graph.get_operations() if op.type == "Conv2D"})
    ops.update({op.name: op for op in graph.get_operations() if op.type == "MatMul"})
    total_loss = (
        total_loss if total_loss is not None else tf_compat.losses.get_total_loss()
    )

    for op_name, op in ops.items():
        _create_parameters_pruning_op(op, _ks_group(op))

    progress = KSLossSensitivityProgress(
        layer_index=-1,
        layer_name="",
        layers=list(ops.keys()),
        sparsity_index=-1,
        sparsity_levels=sparsity_levels,
        measurement_step=-1,
        samples_per_measurement=samples_per_measurement,
    )

    analysis = KSLossSensitivityAnalysis()
    for operation_index, (name, operation) in enumerate(ops.items()):
        progress = _update_progress(
            progress,
            _AnalysisStage.START_OPERATION,
            layer_index=operation_index,
            layer_name=name,
        )
        if progress_hook:
            progress_hook(progress)

        sparsities_loss = []
        for sparsity_index, sparsity_level in enumerate(sparsity_levels):
            progress = _update_progress(
                progress,
                _AnalysisStage.START_SPARSITY_LEVEL,
                sparsity_index=sparsity_index,
            )
            if progress_hook:
                progress_hook(progress)

            # Update the mask of the current operation, reset all others to zero sparsity
            _update_pruning_masks(session, ops, operation, sparsity_level)

            # Run inference, collecting losses across batches
            measurement_step = 0
            results = []
            for idx in range(len(X) // batch_size):
                start_idx = idx * batch_size
                end_idx = min(len(X) - 1, start_idx + batch_size)
                measurement_step += end_idx - start_idx
                if measurement_step > samples_per_measurement:
                    break
                X_batch, y_batch = X[start_idx:end_idx], y[start_idx:end_idx]
                loss = total_loss.eval(
                    feed_dict={X_placeholder: X_batch, y_placeholder: y_batch}
                )
                results.append(loss)
                progress = _update_progress(
                    progress,
                    _AnalysisStage.END_BATCH,
                    measurement_step=measurement_step,
                )
                if progress_hook:
                    progress_hook(progress)
            sparsities_loss.append((sparsity_level, numpy.mean(results)))
        analysis.results.append(
            KSLossSensitivityResult(name, "weight", operation.type, sparsities_loss)
        )
    # Reset all masks to zero sparsity level
    _update_pruning_masks(session, ops)

    return analysis


def _ks_group(op: tf.Operation) -> str:
    """ Name of the sensitivity related ops/variables for a given operation """
    return "{}/sensitivity".format(op.name)


def _update_progress(
    progress: KSLossSensitivityProgress, stage: _AnalysisStage, **kwargs
):
    """
    Update the progress information based on the stage of the analysis

    :param progress: The object holding the analysis progress information
    :param stage: The stage of the sensitivity analysis

    :return The updated progress object
    """

    def _check_stage_info(keys: Union[str, List[str]]):
        if isinstance(keys, str):
            _check_stage_info([keys])
        else:
            for key in keys:
                if key not in kwargs:
                    raise ValueError("{} required".format(key))

    if stage == _AnalysisStage.START_OPERATION:
        _check_stage_info(["layer_index", "layer_name"])
        progress.layer_index = kwargs["layer_index"]
        progress.layer_name = kwargs["layer_name"]
        progress.sparsity_index = -1
        progress.measurement_step = -1
    elif stage == _AnalysisStage.START_SPARSITY_LEVEL:
        _check_stage_info("sparsity_index")
        progress.sparsity_index = kwargs["sparsity_index"]
        progress.measurement_step = -1
    elif stage == _AnalysisStage.END_BATCH:
        _check_stage_info("measurement_step")
        progress.measurement_step = kwargs["measurement_step"]
    return progress


def _create_parameters_pruning_op(op: tf_compat.Operation, ks_group: str):
    """
    Create the mask and operation to update it, used for pruning purposes

    :param op: The current operation whose weights need to be pruned
    :param ks_group: Name of the group under which the new mask and update operation are created

    :return Operations to assign values to the mask
    """
    op_sgv = ge.sgv(op)
    op_var_tens = get_op_input_var(op)

    sparsity = tf_compat.placeholder(
        tf.float32, shape=(), name="{}/sparsity".format(ks_group)
    )

    with tf_compat.variable_scope(ks_group):
        mask = tf_compat.get_variable(
            KSScope.VAR_MASK,
            op_var_tens.get_shape(),
            initializer=tf_compat.ones_initializer(),
            trainable=False,
            dtype=op_var_tens.dtype,
        )

        # create the masked operation and assign as the new input to the op
        masked = tf_compat.math.multiply(mask, op_var_tens, KSScope.OP_MASKED_VAR)
        op_swapped_inputs = [
            inp if inp != op_var_tens else masked for inp in op_sgv.inputs
        ]
        ge.swap_inputs(op, op_swapped_inputs)

        # create the update ops using the target sparsity tensor
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
            tf_compat.greater(abs_var, new_threshold), tf_compat.dtypes.float32
        )
        mask_assign = tf_compat.assign(mask, new_mask, name=KSScope.OP_MASK_ASSIGN)

    # add return state to collections
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OPS_SPARSITY), sparsity
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_MASK), mask
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASKED_VAR), masked
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_ASSIGN), mask_assign
    )

    return mask_assign


def _sparsity_feed_dict(
    ops: Dict[str, tf.Operation],
    current_op: Optional[tf.Operation] = None,
    current_sparsity: float = 0,
):
    """
    Create feed_dict for all sparsity placeholders, used to run the mask assign operations
    If the current operation and sparsity are provided, then the sparsity is used as value to the
    corresponding mask assign, and all the other sparsity placeholders are reset to zeros.

    :param ops: Dictionary of operations
    :param current_op: Current operation to set custom sparsity; if None then all sparsities are zeros
    :param current_sparsity: Sparsity of the current operation if specified; ignored otherwise

    :return A feed dict to be used for running mask assigns
    """
    feed_dict = {}
    for (op_name, op) in ops.items():
        ks_group = _ks_group(op)
        sparsity_holders = tf_compat.get_collection(
            KSScope.collection_name(ks_group, KSScope.OPS_SPARSITY)
        )
        assert len(sparsity_holders) == 1
        sparsity = (
            current_sparsity
            if current_op is not None and op_name == current_op.name
            else 0
        )
        feed_dict[sparsity_holders[0]] = sparsity
    return feed_dict


def _update_pruning_masks(
    session: tf_compat.Session,
    ops: Dict[str, tf.Operation],
    current_op: Optional[tf.Operation] = None,
    current_sparsity: float = 0,
):
    """
    Apply mask assign operations to assign values to mask variables. If a current operation is provided
    then its sparsity can be set; otherwise, all sparsities are set to zeros

    :param session: The current session
    :param ops: Dictionary of operations
    :param current_op: Current operation to set custom sparsity; if None then all sparsities are zeros
    :param current_sparsity: Sparsity of the current operation if specified; ignored otherwise

    :return None
    """
    all_mask_assigns = []
    for (op_name, op) in ops.items():
        ks_group = _ks_group(op)
        mask_assigns = tf_compat.get_collection(
            KSScope.collection_name(ks_group, KSScope.OP_MASK_ASSIGN)
        )
        assert len(mask_assigns) == 1
        all_mask_assigns.append(mask_assigns[0])

    sparsity_feed_dict = _sparsity_feed_dict(ops, current_op, current_sparsity)
    session.run(all_mask_assigns, feed_dict=sparsity_feed_dict)
