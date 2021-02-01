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

import os
from typing import Callable

import numpy
import pytest

from sparseml.tensorflow_v1.optim import (
    EXTRAS_KEY_SUMMARIES,
    BlockPruningMaskCreator,
    ConstantPruningModifier,
    GMPruningModifier,
    ScheduledModifierManager,
)
from sparseml.tensorflow_v1.utils import (
    batch_cross_entropy_loss,
    eval_tensor_sparsity,
    tf_compat,
)
from tests.sparseml.tensorflow_v1.helpers import mlp_net
from tests.sparseml.tensorflow_v1.optim.test_modifier import (
    ScheduledModifierTest,
    conv_graph_lambda,
    mlp_graph_lambda,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: ConstantPruningModifier(
                params=["mlp_net/fc1/weight"],
                start_epoch=0.0,
                end_epoch=20.0,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestConstantPruningModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], GMPruningModifier],
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
                            else:
                                assert masked_sparsity == last_sparsities[index]
                                last_sparsities[index] = masked_sparsity

                modifier.complete_graph(graph, sess)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_constant_pruning_yaml():
    params = "__ALL__"
    start_epoch = 5.0
    end_epoch = 15.0
    yaml_str = """
    !ConstantPruningModifier
        params: {params}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
    """.format(
        params=params, start_epoch=start_epoch, end_epoch=end_epoch
    )
    yaml_modifier = ConstantPruningModifier.load_obj(
        yaml_str
    )  # type: ConstantPruningModifier
    serialized_modifier = ConstantPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: ConstantPruningModifier
    obj_modifier = ConstantPruningModifier(
        params=params, start_epoch=start_epoch, end_epoch=end_epoch
    )

    assert isinstance(yaml_modifier, ConstantPruningModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
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
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: GMPruningModifier(
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
            lambda: GMPruningModifier(
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
            lambda: GMPruningModifier(
                params="__ALL__",
                init_sparsity=0.05,
                final_sparsity=0.6,
                start_epoch=5.0,
                end_epoch=25.0,
                update_frequency=1.0,
                mask_type=BlockPruningMaskCreator([4, 1]),
            ),
        ),
        (
            conv_graph_lambda,
            lambda: GMPruningModifier(
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
            lambda: GMPruningModifier(
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
class TestGMPruningModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], GMPruningModifier],
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
                        sess.run(modifier.sparsity)

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
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_gm_pruning_training_with_manager():
    modifier = GMPruningModifier(
        params=["mlp_net/fc1/weight", "mlp_net/fc3/weight"],
        init_sparsity=0.05,
        final_sparsity=0.8,
        start_epoch=2.0,
        end_epoch=7.0,
        update_frequency=1.0,
    )
    sec_modifier = GMPruningModifier(
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
                    sess.run(global_step)

                    sess.run(mod_ops)
                    update_ready_val = sess.run(modifier.update_ready)
                    sess.run(modifier.sparsity)

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
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_gm_pruning_yaml():
    params = "__ALL__"
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    inter_func = "cubic"
    mask_type = "channel"
    leave_enabled = "True"
    yaml_str = """
    !GMPruningModifier
        params: {params}
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        inter_func: {inter_func}
        mask_type: {mask_type}
        leave_enabled: {leave_enabled}
    """.format(
        params=params,
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        inter_func=inter_func,
        mask_type=mask_type,
        leave_enabled=leave_enabled,
    )
    yaml_modifier = GMPruningModifier.load_obj(yaml_str)  # type: GMPruningModifier
    serialized_modifier = GMPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: GMPruningModifier
    obj_modifier = GMPruningModifier(
        params=params,
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        inter_func=inter_func,
        mask_type=mask_type,
        leave_enabled=leave_enabled,
    )

    assert isinstance(yaml_modifier, GMPruningModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
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
    assert (
        str(yaml_modifier.leave_enabled)
        == str(serialized_modifier.leave_enabled)
        == str(obj_modifier.leave_enabled)
    )
