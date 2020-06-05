import pytest

import os

from typing import Callable
import numpy

from neuralmagicML.tensorflow.utils import (
    VAR_INDEX_FROM_TRAINABLE,
    tf_compat,
    eval_tensor_sparsity,
    batch_cross_entropy_loss,
)
from neuralmagicML.tensorflow.recal import (
    ConstantKSModifier,
    GradualKSModifier,
    ScheduledModifierManager,
    EXTRAS_KEY_SUMMARIES,
    DimensionSparsityMaskCreator,
    BlockSparsityMaskCreator,
)

from tests.tensorflow.helpers import mlp_net
from tests.tensorflow.recal.test_modifier import (
    ScheduledModifierTest,
    mlp_graph_lambda,
    conv_graph_lambda,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: ConstantKSModifier(
                params=["mlp_net/fc1/weight"], start_epoch=0.0, end_epoch=20.0,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestConstantKSModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], GradualKSModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()
            step_placeholder = tf_compat.placeholder(dtype=tf_compat.int64, name="step")
            global_assign = global_step.assign(step_placeholder)

            inp = graph.get_tensor_by_name("inp:0")
            out = graph.get_tensor_by_name("out:0")

            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step, graph
            )
            assert len(mod_ops) == 1
            assert mod_ops[0] is not None
            assert len(mod_extras) == 1
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            assert modifier.prune_op_vars
            assert len(modifier.prune_op_vars) > 0
            last_sparsities = [0.0 for _ in range(len(modifier.prune_op_vars))]

            with tf_compat.Session(graph=graph) as sess:
                sess.run(tf_compat.global_variables_initializer())
                modifier.initialize_session(sess)
                step_counter = 0
                inp_arr = numpy.random.random((1, *inp.shape[1:]))

                for epoch in range(int(modifier.end_epoch + 5.0)):
                    for step in range(steps_per_epoch):
                        res = sess.run(out, feed_dict={inp: inp_arr})
                        assert res.sum() > 0

                        step_counter += 1
                        sess.run(
                            global_assign, feed_dict={step_placeholder: step_counter}
                        )
                        sess.run(mod_ops)

                        for index, op_vars in enumerate(modifier.prune_op_vars):
                            mask_sparsity = eval_tensor_sparsity(op_vars.mask)
                            masked_sparsity = eval_tensor_sparsity(op_vars.masked)

                            assert abs(mask_sparsity - masked_sparsity) < 1e-5

                            if epoch < modifier.start_epoch:
                                assert masked_sparsity < 1e-2
                                assert not update_ready_val
                            else:
                                assert masked_sparsity == last_sparsities[index]
                                last_sparsities[index] = masked_sparsity

                modifier.complete_graph(graph, sess)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_constant_ks_yaml():
    params = "__ALL__"
    start_epoch = 5.0
    end_epoch = 15.0
    yaml_str = f"""
    !ConstantKSModifier
        params: {params}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
    """
    yaml_modifier = ConstantKSModifier.load_obj(yaml_str)  # type: ConstantKSModifier
    serialized_modifier = ConstantKSModifier.load_obj(
        str(yaml_modifier)
    )  # type: ConstantKSModifier
    obj_modifier = ConstantKSModifier(
        params=params, start_epoch=start_epoch, end_epoch=end_epoch
    )

    assert isinstance(yaml_modifier, ConstantKSModifier)
    assert (
        yaml_modifier.params
        == serialized_modifier.params
        == obj_modifier.params
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: GradualKSModifier(
                params=["mlp_net/fc1/weight"],
                init_sparsity=0.05,
                final_sparsity=0.8,
                start_epoch=0.0,
                end_epoch=20.0,
                update_frequency=1.0,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: GradualKSModifier(
                params="__ALL__",
                init_sparsity=0.05,
                final_sparsity=0.6,
                start_epoch=5.0,
                end_epoch=25.0,
                update_frequency=1.0,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: GradualKSModifier(
                layers="__ALL__",
                init_sparsity=0.05,
                final_sparsity=0.6,
                start_epoch=5.0,
                end_epoch=25.0,
                update_frequency=1.0,
                mask_type=BlockSparsityMaskCreator([4, 1]),
            ),
        ),
        (
            conv_graph_lambda,
            lambda: GradualKSModifier(
                params="__ALL__",
                init_sparsity=0.05,
                final_sparsity=0.8,
                start_epoch=0.0,
                end_epoch=20.0,
                update_frequency=1.0,
            ),
        ),
        (
            conv_graph_lambda,
            lambda: GradualKSModifier(
                params=["conv_net/conv1/weight"],
                init_sparsity=0.05,
                final_sparsity=0.6,
                start_epoch=5.0,
                end_epoch=25.0,
                update_frequency=1.0,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestGradualKSModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], GradualKSModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()
            step_placeholder = tf_compat.placeholder(dtype=tf_compat.int64, name="step")
            global_assign = global_step.assign(step_placeholder)

            inp = graph.get_tensor_by_name("inp:0")
            out = graph.get_tensor_by_name("out:0")

            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step, graph
            )
            assert len(mod_ops) == 1
            assert mod_ops[0] is not None
            assert len(mod_extras) == 1
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            assert modifier.prune_op_vars
            assert len(modifier.prune_op_vars) > 0
            last_sparsities = [0.0 for _ in range(len(modifier.prune_op_vars))]

            with tf_compat.Session(graph=graph) as sess:
                sess.run(tf_compat.global_variables_initializer())
                modifier.initialize_session(sess)
                step_counter = 0
                inp_arr = numpy.random.random((1, *inp.shape[1:]))

                for epoch in range(int(modifier.end_epoch + 5.0)):
                    for step in range(steps_per_epoch):
                        res = sess.run(out, feed_dict={inp: inp_arr})
                        assert res.sum() > 0

                        step_counter += 1
                        sess.run(
                            global_assign, feed_dict={step_placeholder: step_counter}
                        )

                        sess.run(mod_ops)
                        update_ready_val = sess.run(modifier.update_ready)
                        sparsity_val = sess.run(modifier.sparsity)

                        for index, op_vars in enumerate(modifier.prune_op_vars):
                            mask_sparsity = eval_tensor_sparsity(op_vars.mask)
                            masked_sparsity = eval_tensor_sparsity(op_vars.masked)

                            assert abs(mask_sparsity - masked_sparsity) < 1e-5

                            if epoch < modifier.start_epoch:
                                assert masked_sparsity < 1e-2
                                assert not update_ready_val
                            elif epoch >= modifier.end_epoch:
                                assert (
                                    abs(masked_sparsity - modifier.final_sparsity)
                                    < 1e-2
                                )
                                assert not update_ready_val
                            else:
                                assert masked_sparsity >= last_sparsities[index] - 1e-2
                                last_sparsities[index] = masked_sparsity

                modifier.complete_graph(graph, sess)

                for op_vars in modifier.prune_op_vars:
                    assert (
                        abs(
                            modifier.final_sparsity
                            - eval_tensor_sparsity(op_vars.op_input)
                        )
                        < 1e-2
                    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_gradual_ks_training_with_manager():
    modifier = GradualKSModifier(
        params=["mlp_net/fc1/weight", "mlp_net/fc3/weight"],
        init_sparsity=0.05,
        final_sparsity=0.8,
        start_epoch=2.0,
        end_epoch=7.0,
        update_frequency=1.0,
    )
    sec_modifier = GradualKSModifier(
        params=["mlp_net/fc2/weight"],
        init_sparsity=0.05,
        final_sparsity=0.8,
        start_epoch=2.0,
        end_epoch=7.0,
        update_frequency=1.0,
    )
    manager = ScheduledModifierManager([modifier, sec_modifier])
    steps_per_epoch = 5
    batch_size = 2

    with tf_compat.Graph().as_default() as graph:
        logits, inputs = mlp_net()
        labels = tf_compat.placeholder(tf_compat.float32, [None, *logits.shape[1:]])
        loss = batch_cross_entropy_loss(logits, labels)

        global_step = tf_compat.train.get_or_create_global_step()
        train_op = tf_compat.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss, global_step=global_step
        )

        mod_ops, mod_extras = manager.create_ops(steps_per_epoch)
        last_sparsities = [0.0 for _ in range(len(modifier.prune_op_vars))]

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            manager.initialize_session(sess)
            batch_lab = numpy.random.random((batch_size, *logits.shape[1:]))
            batch_inp = numpy.random.random((batch_size, *inputs.shape[1:]))

            for epoch in range(int(modifier.end_epoch + 2.0)):
                for step in range(steps_per_epoch):
                    sess.run(train_op, feed_dict={inputs: batch_inp, labels: batch_lab})
                    step_counter = sess.run(global_step)

                    sess.run(mod_ops)
                    update_ready_val = sess.run(modifier.update_ready)
                    sparsity_val = sess.run(modifier.sparsity)

                    for index, op_vars in enumerate(modifier.prune_op_vars):
                        mask_sparsity = eval_tensor_sparsity(op_vars.mask)
                        masked_sparsity = eval_tensor_sparsity(op_vars.masked)

                        assert abs(mask_sparsity - masked_sparsity) < 1e-5

                        if epoch < modifier.start_epoch:
                            assert masked_sparsity < 1e-2
                            assert not update_ready_val
                        elif epoch >= modifier.end_epoch:
                            assert abs(masked_sparsity - modifier.final_sparsity) < 1e-2
                            assert not update_ready_val
                        else:
                            assert masked_sparsity >= last_sparsities[index] - 1e-2
                            last_sparsities[index] = masked_sparsity

            manager.complete_graph()

            for op_vars in modifier.prune_op_vars:
                assert (
                    abs(
                        modifier.final_sparsity - eval_tensor_sparsity(op_vars.op_input)
                    )
                    < 1e-2
                )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_gradual_ks_yaml():
    params = "__ALL__"
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    inter_func = "cubic"
    mask_type = "channel"
    yaml_str = f"""
    !GradualKSModifier
        params: {params}
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """
    yaml_modifier = GradualKSModifier.load_obj(yaml_str)  # type: GradualKSModifier
    serialized_modifier = GradualKSModifier.load_obj(
        str(yaml_modifier)
    )  # type: GradualKSModifier
    obj_modifier = GradualKSModifier(
        params=params,
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        inter_func=inter_func,
        mask_type=mask_type,
    )

    assert isinstance(yaml_modifier, GradualKSModifier)
    assert (
        yaml_modifier.params
        == serialized_modifier.params
        == obj_modifier.params
    )
    assert (
        yaml_modifier.init_sparsity
        == serialized_modifier.init_sparsity
        == obj_modifier.init_sparsity
    )
    assert (
        yaml_modifier.final_sparsity
        == serialized_modifier.final_sparsity
        == obj_modifier.final_sparsity
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
    assert (
        str(yaml_modifier.mask_type)
        == str(serialized_modifier.mask_type)
        == str(obj_modifier.mask_type)
    )
