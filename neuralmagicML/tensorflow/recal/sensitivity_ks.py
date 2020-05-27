"""
Sensitivity analysis implementations for kernel sparsity on Graphs against loss funcs.
"""

from typing import Dict, List, Union, Callable
from collections import namedtuple
import numpy

from neuralmagicML.recal import (
    KSLossSensitivityProgress,
    KSLossSensitivityResult,
    KSLossSensitivityAnalysis,
)
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.utils import get_prunable_ops, VAR_INDEX_FROM_TRAINABLE
from neuralmagicML.tensorflow.recal.mask_ks import KSScope, create_op_pruning
from neuralmagicML.tensorflow.recal.sparsity_mask import (
    SparsityMaskCreator,
    load_mask_creator,
)


__all__ = [
    "SparsePruningOpVars",
    "ks_loss_sensitivity_op_vars",
    "one_shot_ks_loss_sensitivity",
]


SparsePruningOpVars = namedtuple("SparsePruningOpVars", ("op_vars", "sparsity"))


def ks_loss_sensitivity_op_vars(
    graph: tf_compat.Graph = None,
    var_index: Union[int, str] = VAR_INDEX_FROM_TRAINABLE,
    mask_type: Union[str, List[int], SparsityMaskCreator] = 'unstructured',
) -> List[SparsePruningOpVars]:
    """
    Edit the graph for to inject pruning ops and vars to allow for a ks loss
    sensitivity analysis.

    Note: this must be run outside of a session for it to take effect.

    :param graph: the graph to inject pruning ops and vars into,
        if not supplied uses get_default_graph()
    :param var_index: the index for how to find the input variables into
        the prunable ops
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :return: the created pruning op vars to be used in one_shot_ks_loss_sensitivity
    """

    if not graph:
        graph = tf_compat.get_default_graph()

    mask_creator = mask_type
    if not isinstance(mask_type, SparsityMaskCreator):
        mask_creator = load_mask_creator(mask_type)

    ks_group = one_shot_ks_loss_sensitivity.__name__
    prunable_ops = get_prunable_ops(graph)
    op_vars = []

    with graph.as_default():
        for op_index, (prune_name, prune_op) in enumerate(prunable_ops):
            with tf_compat.name_scope(
                KSScope.model(prune_op, ks_group, trailing_slash=True)
            ):
                sparsity = tf_compat.placeholder(
                    dtype=tf_compat.float32, name="sparsity_placeholder"
                )
                update = tf_compat.constant(True, tf_compat.bool)

            prune_op_var = create_op_pruning(
                prune_op, var_index, sparsity, update, ks_group, mask_creator,
            )
            op_vars.append(SparsePruningOpVars(prune_op_var, sparsity))

    return op_vars


def one_shot_ks_loss_sensitivity(
    op_vars: List[SparsePruningOpVars],
    loss_tensor: tf_compat.Tensor,
    samples_per_measurement: int,
    add_ops_creator: Callable[[int], List[tf_compat.Tensor]] = None,
    feed_dict_creator: Callable[[int], Dict[str, tf_compat.Tensor]] = None,
    sess: tf_compat.Session = None,
    sparsity_levels: List[int] = None,
    progress_hook: Callable = None,
):
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    It does not retrain, and instead puts the model to eval mode.
    Moves operation by operation to calculate the sensitivity analysis for each and
    resets the previously run layers.
    Subsequent sparsity checks for layers and levels will be much faster.

    Note: this should be run once a session has been created and
    the variables have been created for the model.

    Note: the graph should be recreated for later training as this creates
    extra ops in the graph that should be reused before continuing in the system.

    :param op_vars: the created pruning op vars from ks_loss_sensitivity_op_vars
    :param loss_tensor: the loss tensor in the model to measure for the sensitivity
    :param samples_per_measurement: the number of session.run calls to run through
        for each sparsity level on each layer
    :param add_ops_creator: a callback to create an op/tens list to be run through
        the session for each measurement. Called for each measurement
    :param feed_dict_creator: a callback to create a feed dict to be run through
        the session for each measurement. Called for each measurement
    :param sess: the session to use
    :param sparsity_levels:
    :param progress_hook:
    :return: the sensitivity results for every op that is prunable
    """

    if not sess:
        sess = tf_compat.get_default_session()

    if sparsity_levels is None:
        sparsity_levels = [0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]

    if progress_hook is None:
        progress_hook = KSLossSensitivityProgress.standard_update_hook()

    progress = KSLossSensitivityProgress(
        layer_index=-1,
        layer_name="",
        layers=[var.op_vars.op.name for var in op_vars],
        sparsity_index=-1,
        sparsity_levels=sparsity_levels,
        measurement_step=-1,
        samples_per_measurement=samples_per_measurement,
    )
    analysis = KSLossSensitivityAnalysis()
    sess.run(tf_compat.variables_initializer([var.op_vars.mask for var in op_vars]))

    for op_index, sparse_op_vars in enumerate(op_vars):
        progress.layer_index = op_index
        progress.layer_name = sparse_op_vars.op_vars.op.name
        progress.sparsity_index = -1
        progress.measurement_step = -1
        if progress_hook:
            progress_hook(progress)

        sparsities_loss = []

        for sparsity_index, sparsity_level in enumerate(sparsity_levels):
            progress.sparsity_index = sparsity_index
            progress.measurement_step = -1
            if progress_hook:
                progress_hook(progress)

            sess.run(
                sparse_op_vars.op_vars.update,
                feed_dict={sparse_op_vars.sparsity: sparsity_level},
            )
            measured = []

            for step in range(samples_per_measurement):
                progress.measurement_step = step
                if progress_hook:
                    progress_hook(progress)

                ops = [loss_tensor]
                add_ops = add_ops_creator(step) if add_ops_creator else None
                feed_dict = feed_dict_creator(step) if feed_dict_creator else None

                if add_ops:
                    ops.extend(add_ops)

                values = sess.run(ops, feed_dict=feed_dict)
                loss = values[0]
                measured.append(loss)

            loss_mean = numpy.mean(measured).item()
            sparsities_loss.append((sparsity_level, loss_mean))

        analysis.results.append(
            KSLossSensitivityResult(
                sparse_op_vars.op_vars.op.name,
                sparse_op_vars.op_vars.op_input.name,
                sparse_op_vars.op_vars.op.type,
                sparsities_loss,
            )
        )
        sess.run(
            sparse_op_vars.op_vars.update, feed_dict={sparse_op_vars.sparsity: 0.0}
        )

    return analysis
