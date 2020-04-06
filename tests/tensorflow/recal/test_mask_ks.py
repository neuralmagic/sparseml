import pytest

import numpy

from neuralmagicML.tensorflow.utils import tf_compat, VAR_INDEX_FROM_TRAINABLE
from neuralmagicML.tensorflow.recal import (
    get_or_create_ks_schedule_ops,
    create_op_pruning,
    get_or_create_graph_ops_pruning,
    get_or_create_ks_scheduled_graph_ops,
)


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
def test_create_op_pruning_fc(sparsity_val):
    group = "test-group"
    graph = tf_compat.Graph()

    with graph.as_default():
        inp = tf_compat.placeholder(tf_compat.float32, [None, 64])

        with tf_compat.name_scope("fc"):
            weights = tf_compat.Variable(
                tf_compat.random_normal([64, 64]), name="weights"
            )
            bias = tf_compat.Variable(tf_compat.random_normal([64]), name="bias")
            matmul = tf_compat.matmul(inp, weights, name="matmul")
            add = tf_compat.add(matmul, bias, name="bias_add")
            relu = tf_compat.nn.relu(add, name="relu")

        sparsity = tf_compat.Variable(0.0, dtype=tf_compat.float32, name="sparsity")
        sparsity_placeholder = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        sparsity_assign = sparsity.assign(sparsity_placeholder)
        matmul_op = graph.get_operation_by_name("fc/matmul")
        pruning_op_vars = create_op_pruning(
            matmul_op, VAR_INDEX_FROM_TRAINABLE, sparsity, group
        )

    with tf_compat.Session(graph=graph) as sess:
        sess.run(tf_compat.global_variables_initializer())
        sess.run(sparsity_assign, feed_dict={sparsity_placeholder: sparsity_val})
        sess.run(pruning_op_vars.assign)

        mask_val = sess.run(pruning_op_vars.mask)
        num_nonzeros = numpy.count_nonzero(mask_val)
        calc_density = float(num_nonzeros) / float(mask_val.size)
        calc_sparsity = 1.0 - calc_density
        assert abs(calc_sparsity - sparsity_val) < 1e-3


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
def test_create_op_pruning_conv(sparsity_val: float):
    group = "test-group"
    graph = tf_compat.Graph()

    with graph.as_default():
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

        sparsity = tf_compat.Variable(0.0, dtype=tf_compat.float32, name="sparsity")
        sparsity_placeholder = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        sparsity_assign = sparsity.assign(sparsity_placeholder)
        conv_op = graph.get_operation_by_name("conv/conv")
        pruning_op_vars = create_op_pruning(
            conv_op, VAR_INDEX_FROM_TRAINABLE, sparsity, group
        )

    with tf_compat.Session(graph=graph) as sess:
        sess.run(tf_compat.global_variables_initializer())
        sess.run(sparsity_assign, feed_dict={sparsity_placeholder: sparsity_val})
        sess.run(pruning_op_vars.assign)

        mask_val = sess.run(pruning_op_vars.mask)
        num_nonzeros = numpy.count_nonzero(mask_val)
        calc_density = float(num_nonzeros) / float(mask_val.size)
        calc_sparsity = 1.0 - calc_density
        assert abs(calc_sparsity - sparsity_val) < 1e-3


@pytest.mark.parametrize("sparsity_val", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0])
def test_get_or_create_graph_ops_pruning(sparsity_val: float):
    group = "test-group"
    graph = tf_compat.Graph()

    with graph.as_default():
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

        reshape = tf_compat.reshape(relu, [-1, 8 * 8 * 64])

        with tf_compat.name_scope("fc"):
            weights = tf_compat.Variable(
                tf_compat.random_normal([8 * 8 * 64, 10]), name="weights"
            )
            bias = tf_compat.Variable(tf_compat.random_normal([10]), name="bias")
            matmul = tf_compat.matmul(reshape, weights, name="matmul")
            add = tf_compat.add(matmul, bias, name="bias_add")
            relu = tf_compat.nn.relu(add, name="relu")

        sparsity = tf_compat.Variable(0.0, dtype=tf_compat.float32, name="sparsity")
        sparsity_placeholder = tf_compat.placeholder(
            dtype=tf_compat.float32, name="sparsity_placeholder"
        )
        sparsity_assign = sparsity.assign(sparsity_placeholder)

        pruning_op_vars = get_or_create_graph_ops_pruning(
            graph, ["conv/conv", "fc/matmul"], VAR_INDEX_FROM_TRAINABLE, sparsity, group
        )
        pruning_op_vars_sec = get_or_create_graph_ops_pruning(
            graph, ["conv/conv", "fc/matmul"], VAR_INDEX_FROM_TRAINABLE, sparsity, group
        )

    for op_vars, op_vars_sec in zip(pruning_op_vars, pruning_op_vars_sec):
        assert op_vars.assign == op_vars_sec.assign
        assert op_vars.mask == op_vars_sec.mask
        assert op_vars.thresh == op_vars_sec.thresh
        assert op_vars.masked == op_vars_sec.masked

    with tf_compat.Session(graph=graph) as sess:
        sess.run(tf_compat.global_variables_initializer())
        sess.run(sparsity_assign, feed_dict={sparsity_placeholder: sparsity_val})

        for op_vars in pruning_op_vars:
            sess.run(op_vars.assign)
            mask_val = sess.run(op_vars.mask)
            num_nonzeros = numpy.count_nonzero(mask_val)
            calc_density = float(num_nonzeros) / float(mask_val.size)
            calc_sparsity = 1.0 - calc_density
            assert abs(calc_sparsity - sparsity_val) < 1e-3


@pytest.mark.parametrize(
    "begin_step,end_step,update_step_freq,init_sparsity,final_sparsity,exponent",
    [
        (0, 99, 1, 0.05, 0.8, 1.0),
        (25, 199, 1, 0.05, 0.8, 1.0),
        (0, 99, 5, 0.05, 0.8, 1.0),
        (0, 99, 1, 0.05, 0.8, 3.0),
        (25, 199, 1, 0.05, 0.8, 3.0),
        (0, 99, 5, 0.05, 0.8, 3.0),
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
    graph = tf_compat.Graph()

    with graph.as_default():
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

    with tf_compat.Session(graph=graph) as sess:
        sess.run(global_step.initializer)
        last_update_step = None
        last_update_sparsity = None

        for step in range(end_step + 10):
            sess.run(global_assign, feed_dict={step_placeholder: step})
            update_ready_val = sess.run(update_ready)
            sparsity_val = sess.run(sparsity)

            if step < begin_step:
                assert not update_ready_val
                assert abs(sparsity_val - init_sparsity) < 1e-5
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


@pytest.mark.parametrize(
    "begin_step,end_step,update_step_freq,init_sparsity,final_sparsity,exponent",
    [
        (0, 99, 1, 0.05, 0.8, 1.0),
        (25, 199, 1, 0.05, 0.8, 1.0),
        (0, 99, 5, 0.05, 0.8, 1.0),
        (0, 99, 1, 0.05, 0.8, 3.0),
        (25, 199, 1, 0.05, 0.8, 3.0),
        (0, 99, 5, 0.05, 0.8, 3.0),
    ],
)
def test_get_or_create_ks_scheduled_graph_ops(
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
):
    group = "test-group"
    graph = tf_compat.Graph()

    with graph.as_default():
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

        global_step = tf_compat.train.get_or_create_global_step()
        step_placeholder = tf_compat.placeholder(dtype=tf_compat.int64, name="step")
        global_assign = global_step.assign(step_placeholder)
        update_op, pruning_op_vars = get_or_create_ks_scheduled_graph_ops(
            graph,
            global_step,
            ["conv/conv"],
            VAR_INDEX_FROM_TRAINABLE,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            group,
        )
        update_op_sec, pruning_op_vars_sec = get_or_create_ks_scheduled_graph_ops(
            graph,
            global_step,
            ["conv/conv"],
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

    for op_var, op_var_sec in zip(pruning_op_vars, pruning_op_vars_sec):
        assert op_var.assign == op_var_sec.assign
        assert op_var.mask == op_var_sec.mask
        assert op_var.thresh == op_var_sec.thresh
        assert op_var.masked == op_var_sec.masked

    with tf_compat.Session(graph=graph) as sess:
        sess.run(tf_compat.global_variables_initializer())
        last_update_sparsity = None

        for step in range(end_step + 10):
            sess.run(global_assign, feed_dict={step_placeholder: step})
            sess.run(update_op)

            mask_val = sess.run(pruning_op_vars[0].mask)
            num_nonzeros = numpy.count_nonzero(mask_val)
            density_val = float(num_nonzeros) / float(mask_val.size)
            sparsity_val = 1.0 - density_val

            if step < begin_step:
                assert abs(sparsity_val) < 1e-3
            elif step == begin_step:
                assert abs(sparsity_val - init_sparsity) < 1e-3
                last_update_sparsity = sparsity_val
            elif step == end_step:
                assert abs(sparsity_val - final_sparsity) < 1e-3
                last_update_sparsity = sparsity_val
            elif step > end_step:
                assert abs(sparsity_val - final_sparsity) < 1e-3
            else:
                assert sparsity_val >= last_update_sparsity
                last_update_sparsity = sparsity_val
