import pytest

from typing import Callable, List
import numpy

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    VAR_INDEX_FROM_TRAINABLE,
    eval_tensor_sparsity,
)
from neuralmagicML.tensorflow.recal import (
    get_or_create_ks_schedule_ops,
    create_op_pruning,
    get_or_create_graph_ops_pruning,
    get_or_create_ks_scheduled_graph_ops,
    apply_op_vars_masks,
)

from tests.tensorflow.helpers import mlp_net, conv_net


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
def test_create_op_pruning_fc(sparsity_val):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        inp = tf_compat.placeholder(tf_compat.float32, [None, 64])

        with tf_compat.name_scope("fc"):
            weights = tf_compat.Variable(
                tf_compat.random_normal([64, 64]), name="weights"
            )
            bias = tf_compat.Variable(tf_compat.random_normal([64]), name="bias")
            matmul = tf_compat.matmul(inp, weights, name="matmul")
            add = tf_compat.add(matmul, bias, name="bias_add")
            relu = tf_compat.nn.relu(add, name="relu")

        sparsity = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        update_ready = tf_compat.placeholder(dtype=tf_compat.bool, name="update_ready")

        matmul_op = graph.get_operation_by_name("fc/matmul")
        pruning_op_vars = create_op_pruning(
            matmul_op, VAR_INDEX_FROM_TRAINABLE, sparsity, update_ready, group
        )

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())
            sess.run(
                pruning_op_vars.update,
                feed_dict={sparsity: sparsity_val, update_ready: False},
            )

            mask_sparsity = eval_tensor_sparsity(pruning_op_vars.mask)
            assert mask_sparsity < 1e-3

            masked_sparsity = eval_tensor_sparsity(pruning_op_vars.masked)
            assert masked_sparsity < 1e-3

            sess.run(
                pruning_op_vars.update,
                feed_dict={sparsity: sparsity_val, update_ready: True},
            )

            mask_sparsity = eval_tensor_sparsity(pruning_op_vars.mask)
            assert abs(mask_sparsity - sparsity_val) < 1e-3

            masked_sparsity = eval_tensor_sparsity(pruning_op_vars.masked)
            assert abs(masked_sparsity - sparsity_val) < 1e-3

            res = sess.run(relu, feed_dict={inp: numpy.random.random((4, 64))})
            assert res.sum() > 0.0


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
def test_create_op_pruning_conv(sparsity_val: float):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        inp = tf_compat.placeholder(tf_compat.float32, [None, 8, 8, 64])

        with tf_compat.name_scope("conv"):
            weights = tf_compat.Variable(
                tf_compat.random_normal([3, 3, 64, 64]), name="weights"
            )
            bias = tf_compat.Variable(tf_compat.random_normal([64]), name="bias")
            conv = tf_compat.nn.conv2d(
                inp, weights, strides=[1, 1, 1, 1], padding="SAME", name="conv"
            )
            add = tf_compat.add(conv, bias, name="bias_add")
            relu = tf_compat.nn.relu(add, name="relu")

        sparsity = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        update_ready = tf_compat.placeholder(dtype=tf_compat.bool, name="update_ready")

        conv_op = graph.get_operation_by_name("conv/conv")
        pruning_op_vars = create_op_pruning(
            conv_op, VAR_INDEX_FROM_TRAINABLE, sparsity, update_ready, group
        )

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())
            sess.run(
                pruning_op_vars.update,
                feed_dict={sparsity: sparsity_val, update_ready: False},
            )

            mask_sparsity = eval_tensor_sparsity(pruning_op_vars.mask)
            assert mask_sparsity < 1e-3

            masked_sparsity = eval_tensor_sparsity(pruning_op_vars.masked)
            assert masked_sparsity < 1e-3

            sess.run(
                pruning_op_vars.update,
                feed_dict={sparsity: sparsity_val, update_ready: True},
            )

            mask_sparsity = eval_tensor_sparsity(pruning_op_vars.mask)
            assert abs(mask_sparsity - sparsity_val) < 1e-3

            masked_sparsity = eval_tensor_sparsity(pruning_op_vars.masked)
            assert abs(masked_sparsity - sparsity_val) < 1e-3

            res = sess.run(relu, feed_dict={inp: numpy.random.random((4, 8, 8, 64))})
            assert res.sum() > 0.0


@pytest.mark.parametrize("sparsity_val", [0.6])
def test_create_op_pruning_decrease_sparsity(sparsity_val):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        inp = tf_compat.placeholder(tf_compat.float32, [None, 64])

        with tf_compat.name_scope("fc"):
            weights = tf_compat.Variable(tf_compat.zeros([64, 64]), name="weights")
            bias = tf_compat.Variable(tf_compat.zeros([64]), name="bias")
            matmul = tf_compat.matmul(inp, weights, name="matmul")
            add = tf_compat.add(matmul, bias, name="bias_add")
            relu = tf_compat.nn.relu(add, name="relu")

        sparsity = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        update_ready = tf_compat.placeholder(dtype=tf_compat.bool, name="update_ready")

        matmul_op = graph.get_operation_by_name("fc/matmul")
        pruning_op_vars = create_op_pruning(
            matmul_op, VAR_INDEX_FROM_TRAINABLE, sparsity, update_ready, group
        )

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())
            sess.run(
                pruning_op_vars.update,
                feed_dict={sparsity: sparsity_val, update_ready: True},
            )

            mask_sparsity = eval_tensor_sparsity(pruning_op_vars.mask)
            assert abs(mask_sparsity - sparsity_val) < 1e-3


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
@pytest.mark.parametrize(
    "net_const,inp_arr,ops",
    [
        (
            mlp_net,
            numpy.random.random((4, 16)),
            ["mlp_net/fc1/matmul", "mlp_net/fc2/matmul", "mlp_net/fc3/matmul"],
        ),
        (
            conv_net,
            numpy.random.random((4, 28, 28, 1)),
            ["conv_net/conv1/conv", "conv_net/conv2/conv", "conv_net/mlp/matmul"],
        ),
    ],
)
def test_get_or_create_graph_ops_pruning(
    sparsity_val: float, net_const: Callable, inp_arr: numpy.ndarray, ops: List[str]
):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        out, inp = net_const()
        sparsity = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        update_ready = tf_compat.placeholder(dtype=tf_compat.bool, name="update_ready")
        pruning_op_vars = get_or_create_graph_ops_pruning(
            graph, ops, VAR_INDEX_FROM_TRAINABLE, sparsity, update_ready, group
        )
        pruning_op_vars_sec = get_or_create_graph_ops_pruning(
            graph, ops, VAR_INDEX_FROM_TRAINABLE, sparsity, update_ready, group
        )

        assert len(pruning_op_vars) == len(ops)

        for op_vars, op_vars_sec in zip(pruning_op_vars, pruning_op_vars_sec):
            assert op_vars.op == op_vars_sec.op
            assert op_vars.update == op_vars_sec.update
            assert op_vars.mask == op_vars_sec.mask
            assert op_vars.masked == op_vars_sec.masked

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())

            for op_vars in pruning_op_vars:
                sess.run(
                    op_vars.update,
                    feed_dict={sparsity: sparsity_val, update_ready: False},
                )

                mask_sparsity = eval_tensor_sparsity(op_vars.mask)
                assert mask_sparsity < 1e-2

                masked_sparsity = eval_tensor_sparsity(op_vars.masked)
                assert masked_sparsity < 1e-2

                sess.run(
                    op_vars.update,
                    feed_dict={sparsity: sparsity_val, update_ready: True},
                )

                mask_sparsity = eval_tensor_sparsity(op_vars.mask)
                assert abs(mask_sparsity - sparsity_val) < 1e-2

                masked_sparsity = eval_tensor_sparsity(op_vars.masked)
                assert abs(masked_sparsity - sparsity_val) < 1e-2

                res = sess.run(out, feed_dict={inp: inp_arr})
                assert res.sum() > 0.0


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
@pytest.mark.parametrize(
    "net_const,inp_arr,ops",
    [
        (
            mlp_net,
            numpy.random.random((4, 16)),
            ["mlp_net/fc1/matmul", "mlp_net/fc2/matmul", "mlp_net/fc3/matmul"],
        ),
        (
            conv_net,
            numpy.random.random((4, 28, 28, 1)),
            ["conv_net/conv1/conv", "conv_net/conv2/conv", "conv_net/mlp/matmul"],
        ),
    ],
)
def test_apply_op_vars_masks(
    sparsity_val: float, net_const: Callable, inp_arr: numpy.ndarray, ops: List[str]
):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        out, inp = net_const()
        sparsity = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        update_ready = tf_compat.placeholder(dtype=tf_compat.bool, name="update_ready")
        pruning_op_vars = get_or_create_graph_ops_pruning(
            graph, ops, VAR_INDEX_FROM_TRAINABLE, sparsity, update_ready, group
        )

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())

            for op_vars in pruning_op_vars:
                sess.run(
                    op_vars.update,
                    feed_dict={sparsity: sparsity_val, update_ready: True},
                )

            apply_op_vars_masks(pruning_op_vars, group, sess)

            for op_vars in pruning_op_vars:
                var_sparsity = eval_tensor_sparsity(op_vars.op_input)
                assert abs(var_sparsity - sparsity_val) < 1e-2


@pytest.mark.parametrize(
    "begin_step,end_step,update_step_freq,init_sparsity,final_sparsity,exponent",
    [
        (0, 99, 1, 0.05, 0.8, 1.0),
        (25, 199, 1, 0.05, 0.8, 1.0),
        (0, 99, 5, 0.05, 0.8, 1.0),
        (25, 199, 5, 0.05, 0.8, 1.0),
        (0, 99, 1, 0.05, 0.8, 3.0),
        (25, 199, 1, 0.05, 0.8, 3.0),
        (0, 99, 5, 0.05, 0.8, 3.0),
        (25, 199, 5, 0.05, 0.8, 3.0),
    ],
)
def test_get_or_create_ks_schedule_ops(
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        global_step = tf_compat.train.get_or_create_global_step()
        step_placeholder = tf_compat.placeholder(dtype=tf_compat.int64, name="step")
        global_assign = global_step.assign(step_placeholder)

        update_ready, sparsity = get_or_create_ks_schedule_ops(
            global_step,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            group,
        )
        update_ready_sec, sparsity_sec = get_or_create_ks_schedule_ops(
            global_step,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            group,
        )

        assert update_ready == update_ready_sec
        assert sparsity == sparsity_sec

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())
            last_update_step = None
            last_update_sparsity = None

            for step in range(end_step + 10):
                sess.run(global_assign, feed_dict={step_placeholder: step})
                update_ready_val = sess.run(update_ready)
                sparsity_val = sess.run(sparsity)

                if step < begin_step:
                    assert not update_ready_val
                    assert abs(sparsity_val) < 1e-5
                elif step <= begin_step:
                    assert update_ready_val
                    assert abs(sparsity_val - init_sparsity) < 1e-5
                    last_update_step = step
                    last_update_sparsity = sparsity_val
                elif step == end_step:
                    assert update_ready_val
                    assert abs(sparsity_val - final_sparsity) < 1e-5
                    last_update_step = step
                    last_update_sparsity = sparsity_val
                elif step > end_step:
                    assert not update_ready_val
                    assert abs(sparsity_val - final_sparsity) < 1e-5
                else:
                    # check if update should be ready
                    check_ready = (
                        last_update_step is None
                        or step >= last_update_step + update_step_freq
                    )
                    assert sparsity_val > last_update_sparsity

                    if check_ready:
                        assert update_ready_val
                        last_update_step = step
                        last_update_sparsity = sparsity_val
                    else:
                        assert not update_ready_val


def _expected_sparsity(
    steps_count: int,
    steps_range: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
):
    percentage = steps_count / steps_range
    return init_sparsity + (1 - (1 - percentage) ** exponent) * (
        final_sparsity - init_sparsity
    )


@pytest.mark.parametrize(
    "begin_step,end_step,update_step_freq,init_sparsity,final_sparsity,exponent",
    [
        (0, 99, 1, 0.05, 0.8, 1.0),
        (25, 199, 1, 0.05, 0.8, 1.0),
        (0, 99, 5, 0.05, 0.8, 1.0),
        (25, 199, 5, 0.05, 0.8, 1.0),
        (0, 99, 1, 0.05, 0.8, 3.0),
        (25, 199, 1, 0.05, 0.8, 3.0),
        (0, 99, 5, 0.05, 0.8, 3.0),
        (25, 199, 5, 0.05, 0.8, 3.0),
        (0, 99, 1, 0.8, 0.05, 1.0),
        (25, 199, 5, 0.8, 0.05, 1.0),
    ],
)
@pytest.mark.parametrize(
    "net_const,inp_arr,ops",
    [
        (
            mlp_net,
            numpy.random.random((4, 16)),
            ["mlp_net/fc1/matmul", "mlp_net/fc2/matmul", "mlp_net/fc3/matmul"],
        ),
        (
            conv_net,
            numpy.random.random((4, 28, 28, 1)),
            ["conv_net/conv1/conv", "conv_net/conv2/conv", "conv_net/mlp/matmul"],
        ),
    ],
)
def test_get_or_create_ks_scheduled_graph_ops(
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    net_const: Callable,
    inp_arr: numpy.ndarray,
    ops: List[str],
):
    group = "test-group"

    with tf_compat.Graph().as_default() as graph:
        global_step = tf_compat.train.get_or_create_global_step()
        step_placeholder = tf_compat.placeholder(dtype=tf_compat.int64, name="step")
        global_assign = global_step.assign(step_placeholder)

        out, inp = net_const()

        (
            update_op,
            pruning_op_vars,
            update_ready,
            sparsity,
        ) = get_or_create_ks_scheduled_graph_ops(
            graph,
            global_step,
            ops,
            VAR_INDEX_FROM_TRAINABLE,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            group,
        )
        (
            update_op_sec,
            pruning_op_vars_sec,
            update_ready,
            sparsity,
        ) = get_or_create_ks_scheduled_graph_ops(
            graph,
            global_step,
            ops,
            VAR_INDEX_FROM_TRAINABLE,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            group,
        )

        assert update_op == update_op_sec
        assert update_ready == update_ready
        assert sparsity == sparsity
        assert len(pruning_op_vars) == len(ops)

        for op_vars, op_vars_sec in zip(pruning_op_vars, pruning_op_vars_sec):
            assert op_vars.op == op_vars_sec.op
            assert op_vars.update == op_vars_sec.update
            assert op_vars.mask == op_vars_sec.mask
            assert op_vars.masked == op_vars_sec.masked

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())
            last_update_sparsity = None

            for step in range(end_step + 10):
                sess.run(global_assign, feed_dict={step_placeholder: step})
                update_ready_val = sess.run(update_ready)
                sparsity_val = sess.run(sparsity)
                sess.run(update_op)

                for op_var in pruning_op_vars:
                    mask_sparsity = eval_tensor_sparsity(op_var.mask)
                    masked_sparsity = eval_tensor_sparsity(op_var.masked)

                    assert abs(mask_sparsity - masked_sparsity) < 1e-5

                    if step < begin_step:
                        assert abs(masked_sparsity) < 1e-2
                        assert not update_ready_val
                    elif step == begin_step:
                        assert abs(masked_sparsity - init_sparsity) < 1e-2
                        assert abs(sparsity_val - init_sparsity) < 1e-5
                        assert update_ready_val
                        last_update_sparsity = masked_sparsity
                    elif step == end_step:
                        assert update_ready_val
                        assert abs(masked_sparsity - final_sparsity) < 1e-2
                        assert abs(sparsity_val - final_sparsity) < 1e-5
                        last_update_sparsity = masked_sparsity
                    elif step > end_step:
                        assert not update_ready_val
                        assert abs(masked_sparsity - final_sparsity) < 1e-2
                    elif init_sparsity > final_sparsity:
                        assert masked_sparsity <= last_update_sparsity + 1e-2
                        assert sparsity_val <= last_update_sparsity + 1e-2
                        last_update_sparsity = masked_sparsity
                    else:
                        assert masked_sparsity >= last_update_sparsity - 1e-2
                        assert sparsity_val >= last_update_sparsity - 1e-2
                        last_update_sparsity = masked_sparsity
                        if step < end_step and update_ready_val:
                            steps_count = sess.run(global_step) - begin_step
                            steps_range = end_step - begin_step
                            expected = _expected_sparsity(
                                steps_count,
                                steps_range,
                                init_sparsity,
                                final_sparsity,
                                exponent,
                            )
                            assert abs(sparsity_val - expected) < 1e-5

                res = sess.run(out, feed_dict={inp: inp_arr})
                assert res.sum() >= 0.0
