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

import math
import os
from typing import Callable

import numpy as np
import pytest

from sparseml.tensorflow_v1.optim import (
    GroupLearningRateModifier,
    LearningRateModifier,
    Modifier,
    ScheduledModifierManager,
    SetLearningRateModifier,
)
from sparseml.tensorflow_v1.optim.modifier import (
    EXTRAS_KEY_LEARNING_RATE,
    EXTRAS_KEY_SUMMARIES,
)
from sparseml.tensorflow_v1.utils import tf_compat
from tests.sparseml.tensorflow_v1.optim.test_modifier import (
    ScheduledModifierTest,
    mlp_graph_lambda,
)


EPSILON = 1e-7


##############################
#
# SetLearningRateModifier tests
#
##############################


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: SetLearningRateModifier(learning_rate=0.1),
        ),
        (
            mlp_graph_lambda,
            lambda: SetLearningRateModifier(learning_rate=0.03, start_epoch=5),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestSetLRModifierImpl(ScheduledModifierTest):
    @pytest.mark.parametrize(
        "optim_lambda",
        [tf_compat.train.GradientDescentOptimizer, tf_compat.train.AdamOptimizer],
    )
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], SetLearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
        optim_lambda,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()

        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            x_batch = graph.get_tensor_by_name("inp:0")
            y_pred = graph.get_tensor_by_name("out:0")
            n_inputs = x_batch.shape[1]
            n_outputs = y_pred.shape[1]
            y_lab = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(mod_ops) == 0
            assert len(mod_extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in mod_extras
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

            with tf_compat.name_scope("train"):
                optimizer = optim_lambda(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_lab, y_pred)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        batch_size = 8
        batch_x = np.random.randn(batch_size, n_inputs)
        batch_lab = np.random.randn(batch_size, n_outputs)

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            for epoch in range(int(max(modifier.start_epoch, modifier.end_epoch)) + 5):
                for step in range(steps_per_epoch):
                    gs = sess.run(global_step)
                    expected = modifier.learning_rate
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert (
                        abs(optim_lr - expected) <= EPSILON
                    ), "Failed at epoch:{} step:{} global_step:{}".format(
                        epoch, step, gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={x_batch: batch_x, y_lab: batch_lab},
                    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_set_lr_yaml():
    start_epoch = 10.0
    set_lr = 0.1
    yaml_str = """
    !SetLearningRateModifier
        learning_rate: {}
        start_epoch: {}
    """.format(
        set_lr, start_epoch
    )
    yaml_modifier = SetLearningRateModifier.load_obj(
        yaml_str
    )  # type: SetLearningRateModifier
    serialized_modifier = SetLearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: SetLearningRateModifier
    obj_modifier = SetLearningRateModifier(
        learning_rate=set_lr, start_epoch=start_epoch
    )

    assert isinstance(yaml_modifier, SetLearningRateModifier)
    assert (
        yaml_modifier.learning_rate
        == serialized_modifier.learning_rate
        == obj_modifier.learning_rate
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )


##############################
#
# LearningRateModifier tests
#
##############################


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="ExponentialLR",
                lr_kwargs={"gamma": 0.9},
                start_epoch=0,
                init_lr=0.1,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="ExponentialLR",
                lr_kwargs={"gamma": 0.5},
                start_epoch=5,
                end_epoch=13,
                init_lr=0.1,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestLRModifierExponentialImpl(ScheduledModifierTest):
    @pytest.mark.parametrize(
        "optim_lambda",
        [tf_compat.train.GradientDescentOptimizer, tf_compat.train.AdamOptimizer],
    )
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], SetLearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
        optim_lambda,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()

        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            x_batch = graph.get_tensor_by_name("inp:0")
            y_pred = graph.get_tensor_by_name("out:0")
            n_inputs = x_batch.shape[1]
            n_outputs = y_pred.shape[1]
            y_lab = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(mod_ops) == 0
            assert len(mod_extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in mod_extras
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

            with tf_compat.name_scope("train"):
                optimizer = optim_lambda(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_lab, y_pred)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        batch_size = 8
        batch_x = np.random.randn(batch_size, n_inputs)
        batch_lab = np.random.randn(batch_size, n_outputs)

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())

            for epoch in range(int(max(modifier.start_epoch, modifier.end_epoch)) + 5):
                if epoch < modifier.start_epoch:
                    expected = modifier.init_lr
                elif epoch < modifier.end_epoch or modifier.end_epoch == -1:
                    expected = modifier.init_lr * (
                        modifier.lr_kwargs["gamma"]
                        ** (epoch - int(modifier.start_epoch))
                    )
                else:
                    expected = modifier.init_lr * (
                        modifier.lr_kwargs["gamma"]
                        ** int(modifier.end_epoch - modifier.start_epoch - 1)
                    )

                for step in range(steps_per_epoch):
                    gs = sess.run(global_step)
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert (
                        abs(optim_lr - expected) <= EPSILON
                    ), "Failed at epoch:{} step:{} global_step:{}".format(
                        epoch, step, gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={x_batch: batch_x, y_lab: batch_lab},
                    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_lr_modifier_exponential_yaml():
    gamma = 0.9
    lr_class = "ExponentialLR"
    lr_kwargs = {"gamma": gamma}
    start_epoch = 10.0
    end_epoch = 20.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, lr_class, lr_kwargs, init_lr
    )
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        lr_class=lr_class,
        lr_kwargs=lr_kwargs,
        init_lr=init_lr,
    )

    assert isinstance(yaml_modifier, LearningRateModifier)
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
        yaml_modifier.lr_class == serialized_modifier.lr_class == obj_modifier.lr_class
    )
    assert (
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="StepLR",
                lr_kwargs={"gamma": 0.9, "step_size": 3},
                start_epoch=0,
                init_lr=0.1,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="StepLR",
                lr_kwargs={"gamma": 0.5, "step_size": 2},
                start_epoch=5,
                end_epoch=11,
                init_lr=0.01,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestLRModifierStepImpl(ScheduledModifierTest):
    @pytest.mark.parametrize(
        "optim_lambda",
        [tf_compat.train.GradientDescentOptimizer, tf_compat.train.AdamOptimizer],
    )
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], SetLearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
        optim_lambda,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()

        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            x_batch = graph.get_tensor_by_name("inp:0")
            y_pred = graph.get_tensor_by_name("out:0")
            n_inputs = x_batch.shape[1]
            n_outputs = y_pred.shape[1]
            y_lab = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(mod_ops) == 0
            assert len(mod_extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in mod_extras
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

            with tf_compat.name_scope("train"):
                optimizer = optim_lambda(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_lab, y_pred)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        batch_size = 8
        batch_x = np.random.randn(batch_size, n_inputs)
        batch_lab = np.random.randn(batch_size, n_outputs)

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())

            for epoch in range(int(max(modifier.start_epoch, modifier.end_epoch)) + 5):
                if epoch < modifier.start_epoch:
                    expected = modifier.init_lr
                elif epoch < modifier.end_epoch or modifier.end_epoch == -1:
                    expected = modifier.init_lr * (
                        modifier.lr_kwargs["gamma"]
                        ** math.floor(
                            (epoch - modifier.start_epoch)
                            / modifier.lr_kwargs["step_size"]
                        )
                    )
                else:
                    expected = modifier.init_lr * (
                        modifier.lr_kwargs["gamma"]
                        ** (
                            math.floor(
                                (modifier.end_epoch - modifier.start_epoch)
                                / modifier.lr_kwargs["step_size"]
                            )
                            - 1
                        )
                    )

                for step in range(steps_per_epoch):
                    gs = sess.run(global_step)
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert (
                        abs(optim_lr - expected) <= EPSILON
                    ), "Failed at epoch:{} step:{} global_step:{}".format(
                        epoch, step, gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={x_batch: batch_x, y_lab: batch_lab},
                    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_lr_modifier_step_yaml():
    gamma = 0.9
    lr_class = "StepLR"
    lr_kwargs = {"step_size": 1.0, "gamma": gamma}
    start_epoch = 10.0
    end_epoch = 20.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, lr_class, lr_kwargs, init_lr
    )
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        lr_class=lr_class,
        lr_kwargs=lr_kwargs,
        init_lr=init_lr,
    )

    assert isinstance(yaml_modifier, LearningRateModifier)
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
        yaml_modifier.lr_class == serialized_modifier.lr_class == obj_modifier.lr_class
    )
    assert (
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="MultiStepLR",
                lr_kwargs={"gamma": 0.9, "milestones": [1, 3, 4]},
                start_epoch=0,
                init_lr=0.1,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="MultiStepLR",
                lr_kwargs={"gamma": 0.95, "milestones": [5, 8]},
                start_epoch=3,
                end_epoch=13,
                init_lr=0.1,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestLRModifierMultiStepImpl(ScheduledModifierTest):
    @pytest.mark.parametrize(
        "optim_lambda",
        [tf_compat.train.GradientDescentOptimizer, tf_compat.train.AdamOptimizer],
    )
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], LearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
        optim_lambda,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()

        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            x_batch = graph.get_tensor_by_name("inp:0")
            y_pred = graph.get_tensor_by_name("out:0")
            n_inputs = x_batch.shape[1]
            n_outputs = y_pred.shape[1]
            y_lab = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(mod_ops) == 0
            assert len(mod_extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in mod_extras
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

            with tf_compat.name_scope("train"):
                optimizer = optim_lambda(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_lab, y_pred)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        batch_size = 8
        batch_x = np.random.randn(batch_size, n_inputs)
        batch_lab = np.random.randn(batch_size, n_outputs)

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())

            for epoch in range(int(max(modifier.start_epoch, modifier.end_epoch)) + 5):
                if epoch < modifier.start_epoch:
                    expected = modifier.init_lr
                else:
                    num_gammas = sum(
                        [
                            1
                            for mile in modifier.lr_kwargs["milestones"]
                            if epoch >= mile
                        ]
                    )
                    expected = (
                        modifier.init_lr * modifier.lr_kwargs["gamma"] ** num_gammas
                    )

                for step in range(steps_per_epoch):
                    gs = sess.run(global_step)
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert (
                        abs(optim_lr - expected) <= EPSILON
                    ), "Failed at epoch:{} step:{} global_step:{}".format(
                        epoch, step, gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={x_batch: batch_x, y_lab: batch_lab},
                    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_lr_modifier_multi_step_yaml():
    milestones = [1, 3, 4]
    gamma = 0.9
    lr_class = "MultiStepLR"
    lr_kwargs = {"milestones": milestones, "gamma": gamma}
    start_epoch = 2.0
    end_epoch = 20.0
    init_lr = 0.1
    yaml_str = """
    !LearningRateModifier
        start_epoch: {}
        end_epoch: {}
        lr_class: {}
        lr_kwargs: {}
        init_lr: {}
    """.format(
        start_epoch, end_epoch, lr_class, lr_kwargs, init_lr
    )
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        lr_class=lr_class,
        lr_kwargs=lr_kwargs,
        init_lr=init_lr,
    )

    assert isinstance(yaml_modifier, LearningRateModifier)
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
        yaml_modifier.lr_class == serialized_modifier.lr_class == obj_modifier.lr_class
    )
    assert (
        yaml_modifier.lr_kwargs
        == serialized_modifier.lr_kwargs
        == obj_modifier.lr_kwargs
    )
    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


##############################
#
# GroupLearningRateModifier tests
#
##############################


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: GroupLearningRateModifier(
                [
                    SetLearningRateModifier(learning_rate=0.1, start_epoch=0),
                    LearningRateModifier(
                        lr_class="ExponentialLR",
                        lr_kwargs={"gamma": 0.9},
                        start_epoch=5,
                        init_lr=0.01,
                    ),
                ]
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestGroupLearningRateImpl(ScheduledModifierTest):
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        # ignore test, group does not support yaml
        pass

    def test_yaml_key(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        # ignore test, group does not support yaml
        pass

    @pytest.mark.parametrize(
        "optim_lambda",
        [tf_compat.train.GradientDescentOptimizer, tf_compat.train.AdamOptimizer],
    )
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], LearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
        optim_lambda,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()

        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            x_batch = graph.get_tensor_by_name("inp:0")
            y_pred = graph.get_tensor_by_name("out:0")
            n_inputs = x_batch.shape[1]
            n_outputs = y_pred.shape[1]
            y_lab = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(mod_ops) == 0
            assert len(mod_extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in mod_extras
            assert EXTRAS_KEY_SUMMARIES in mod_extras
            learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

            with tf_compat.name_scope("train"):
                optimizer = optim_lambda(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_lab, y_pred)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        batch_size = 8
        batch_x = np.random.randn(batch_size, n_inputs)
        batch_lab = np.random.randn(batch_size, n_outputs)

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())

            for epoch in range(int(max(modifier.start_epoch, modifier.end_epoch)) + 10):
                # for now hardcoding the tests to get out the door
                if epoch < 5:
                    expected = 0.1
                else:
                    expected = 0.01 * 0.9 ** (epoch - 5)

                for step in range(steps_per_epoch):
                    gs = sess.run(global_step)
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert (
                        abs(optim_lr - expected) <= EPSILON
                    ), "Failed at epoch:{} step:{} global_step:{}".format(
                        epoch, step, gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={x_batch: batch_x, y_lab: batch_lab},
                    )


##############################
#
# LR Manager tests
#
##############################


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "optim_lambda",
    [tf_compat.train.GradientDescentOptimizer, tf_compat.train.AdamOptimizer],
)
def test_lrs_with_manager(optim_lambda):
    manager = ScheduledModifierManager(
        modifiers=[
            SetLearningRateModifier(learning_rate=0.1, start_epoch=0),
            LearningRateModifier(
                lr_class="ExponentialLR",
                lr_kwargs={"gamma": 0.9},
                start_epoch=5,
                end_epoch=10,
                init_lr=0.01,
            ),
            LearningRateModifier(
                lr_class="MultiStepLR",
                lr_kwargs={"gamma": 0.95, "milestones": [15, 18]},
                start_epoch=12,
                end_epoch=20,
                init_lr=0.05,
            ),
        ]
    )
    assert manager.max_epochs == 20
    assert manager.min_epochs == 0
    graph = mlp_graph_lambda()
    steps_per_epoch = 100

    with graph.as_default():
        global_step = tf_compat.train.get_or_create_global_step()

        # Further set up for loss, optimizer and training op
        x_batch = graph.get_tensor_by_name("inp:0")
        y_pred = graph.get_tensor_by_name("out:0")
        n_inputs = x_batch.shape[1]
        n_outputs = y_pred.shape[1]
        y_lab = tf_compat.placeholder(
            tf_compat.float32, shape=(None, n_outputs), name="y"
        )
        mod_ops, mod_extras = manager.create_ops(
            steps_per_epoch, global_step=global_step, graph=graph
        )
        assert len(mod_ops) == 1
        assert len(mod_extras) == 2
        assert EXTRAS_KEY_LEARNING_RATE in mod_extras
        assert EXTRAS_KEY_SUMMARIES in mod_extras
        assert isinstance(mod_extras[EXTRAS_KEY_SUMMARIES], list)
        assert len(mod_extras[EXTRAS_KEY_SUMMARIES]) == 1
        learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

        with tf_compat.name_scope("train"):
            optimizer = optim_lambda(learning_rate=learning_rate)
            loss = tf_compat.losses.mean_squared_error(y_lab, y_pred)
            training_op = optimizer.minimize(loss, global_step=global_step)

    np.random.seed(12)
    batch_size = 8
    batch_x = np.random.randn(batch_size, n_inputs)
    batch_lab = np.random.randn(batch_size, n_outputs)

    with tf_compat.Session(graph=graph) as sess:
        sess.run(tf_compat.global_variables_initializer())

        for epoch in range(manager.max_epochs + 5):
            # for now hardcoding the tests to get out the door
            if epoch < 5:
                expected = 0.1
            elif epoch < 10:
                expected = 0.01 * 0.9 ** (epoch - 5)
            elif epoch < 12:
                expected = 0.01 * 0.9**4
            elif epoch < 15:
                expected = 0.05
            elif epoch < 18:
                expected = 0.05 * 0.95
            else:
                expected = 0.05 * 0.95**2

            for step in range(steps_per_epoch):
                gs = sess.run(global_step)
                optim_lr = sess.run(_get_lr(optimizer))
                assert (
                    abs(optim_lr - expected) <= EPSILON
                ), "Failed at epoch:{} step:{} global_step:{}".format(epoch, step, gs)
                sess.run(
                    training_op,
                    feed_dict={x_batch: batch_x, y_lab: batch_lab},
                )


def _get_lr(optim) -> tf_compat.Variable:
    if hasattr(optim, "_learning_rate"):
        return optim._learning_rate
    if hasattr(optim, "_lr"):
        return optim._lr
    raise ValueError("Internal LR not found")
