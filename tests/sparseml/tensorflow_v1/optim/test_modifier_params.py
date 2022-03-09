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
    EXTRAS_KEY_VAR_LIST,
    ScheduledModifierManager,
    TrainableParamsModifier,
)
from sparseml.tensorflow_v1.utils import batch_cross_entropy_loss, tf_compat
from sparseml.utils import ALL_TOKEN
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
            lambda: TrainableParamsModifier(
                params=["mlp_net/fc1/weight"],
                trainable=False,
                params_strict=True,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: TrainableParamsModifier(
                params=ALL_TOKEN,
                trainable=False,
                params_strict=False,
            ),
        ),
        (
            conv_graph_lambda,
            lambda: TrainableParamsModifier(
                params=["conv_net/conv1/bias"],
                trainable=False,
                params_strict=True,
            ),
        ),
        (
            conv_graph_lambda,
            lambda: TrainableParamsModifier(
                params=["re:.*weight", "re:.*bias"],
                trainable=False,
                params_strict=False,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestTrainableParamsModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], TrainableParamsModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            _, mod_extras = modifier.create_ops(steps_per_epoch, global_step, graph)
            assert len(mod_extras) == 1
            assert EXTRAS_KEY_VAR_LIST in mod_extras
            var_list = mod_extras[EXTRAS_KEY_VAR_LIST]
            assert len(var_list) > 0
            for var in var_list:
                if modifier._trainable:
                    assert var in tf_compat.trainable_variables()
                else:
                    assert var not in tf_compat.trainable_variables()

            with tf_compat.Session(graph=graph) as sess:
                # No ops to invoke

                modifier.complete_graph(graph, sess)
                for var, is_var_trainable in modifier._vars_to_trainable_orig.items():
                    if is_var_trainable:
                        assert var in tf_compat.trainable_variables()
                    else:
                        assert var not in tf_compat.trainable_variables()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_trainable_params_modifier_with_training():
    modifier = TrainableParamsModifier(
        params=["mlp_net/fc1/weight"],
        trainable=False,
        params_strict=False,
    )
    manager = ScheduledModifierManager([modifier])
    steps_per_epoch = 5
    batch_size = 2

    with tf_compat.Graph().as_default() as graph:
        logits, inputs = mlp_net()
        labels = tf_compat.placeholder(tf_compat.float32, [None, *logits.shape[1:]])
        loss = batch_cross_entropy_loss(logits, labels)

        global_step = tf_compat.train.get_or_create_global_step()
        num_trainable_variabls_init = len(tf_compat.trainable_variables())

        mod_ops, mod_extras = manager.create_ops(steps_per_epoch)
        assert len(tf_compat.trainable_variables()) < num_trainable_variabls_init
        # Get the variables returned by the trainable_params modifier
        non_trainable_vars = mod_extras[EXTRAS_KEY_VAR_LIST]
        trainable_vars = tf_compat.trainable_variables()
        train_op = tf_compat.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss, global_step=global_step
        )

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            manager.initialize_session(sess)
            init_non_trainable_vars = [
                var.eval(session=sess) for var in non_trainable_vars
            ]
            init_trainable_vars = [var.eval(session=sess) for var in trainable_vars]
            batch_lab = numpy.random.random((batch_size, *logits.shape[1:]))
            batch_inp = numpy.random.random((batch_size, *inputs.shape[1:]))

            for epoch in range(10):
                for step in range(steps_per_epoch):
                    sess.run(train_op, feed_dict={inputs: batch_inp, labels: batch_lab})
                    sess.run(global_step)
            # Compare initial and final variable values
            for idx, init_non_trainable_var in enumerate(init_non_trainable_vars):
                final_non_trainable_var = non_trainable_vars[idx].eval(session=sess)
                assert numpy.array_equal(
                    init_non_trainable_var, final_non_trainable_var
                )
            for idx, init_trainable_var in enumerate(init_trainable_vars):
                final_trainable_var = trainable_vars[idx].eval(session=sess)
                assert not numpy.array_equal(init_trainable_var, final_trainable_var)
            manager.complete_graph()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_trainable_params_yaml():
    params = ALL_TOKEN
    trainable = False
    params_strict = False
    yaml_str = """
    !TrainableParamsModifier
        params: {params}
        trainable: {trainable}
        params_strict: {params_strict}
    """.format(
        params=params, trainable=trainable, params_strict=params_strict
    )
    yaml_modifier = TrainableParamsModifier.load_obj(
        yaml_str
    )  # type: TrainableParamsModifier
    serialized_modifier = TrainableParamsModifier.load_obj(
        str(yaml_modifier)
    )  # type: TrainableParamsModifier
    obj_modifier = TrainableParamsModifier(
        params=params, trainable=trainable, params_strict=params_strict
    )

    assert isinstance(yaml_modifier, TrainableParamsModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert (
        yaml_modifier.trainable
        == serialized_modifier.trainable
        == obj_modifier.trainable
    )
    assert (
        yaml_modifier.params_strict
        == serialized_modifier.params_strict
        == obj_modifier.params_strict
    )
