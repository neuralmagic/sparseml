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
from typing import Callable, Dict, List

import pytest

from sparseml.tensorflow_v1.optim import (
    TENSORFLOW_V1_FRAMEWORK,
    Modifier,
    ScheduledModifier,
    ScheduledUpdateModifier,
    TensorFlowModifierYAML,
)
from sparseml.tensorflow_v1.utils import tf_compat
from tests.sparseml.optim.test_modifier import (
    BaseModifierTest,
    BaseScheduledTest,
    BaseUpdateTest,
)
from tests.sparseml.tensorflow_v1.helpers import conv_net, mlp_net


def mlp_graph_lambda():
    graph = tf_compat.Graph()

    with graph.as_default():
        mlp_net()

    return graph


def conv_graph_lambda():
    graph = tf_compat.Graph()

    with graph.as_default():
        conv_net()

    return graph


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
class ModifierTest(BaseModifierTest):
    # noinspection PyMethodOverriding
    def test_constructor(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_constructor(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_yaml(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml_key(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_yaml_key(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_repr(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_repr(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)

    def test_create_ops(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()
            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step, graph
            )

        assert mod_ops is not None
        assert isinstance(mod_ops, List)
        assert mod_extras is not None
        assert isinstance(mod_extras, Dict)
        assert modifier.initialized

    def test_modify_estimator(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        def model_fn(features, labels, mode, params):
            graph_lambda()
            return tf_compat.estimator.EstimatorSpec(mode)

        modifier = modifier_lambda()
        tf_compat.get_default_graph()
        estimator = tf_compat.estimator.Estimator(
            model_fn=model_fn,
        )
        assert estimator._model_fn == model_fn
        modifier.modify_estimator(estimator, steps_per_epoch)
        assert estimator._model_fn != model_fn

    def test_initialize_session(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            with tf_compat.Session() as sess:
                with pytest.raises(RuntimeError):
                    modifier.initialize_session(sess)

            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step, graph
            )

            with tf_compat.Session() as sess:
                modifier.initialize_session(sess)

    def test_complete_graph(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            with pytest.raises(RuntimeError):
                modifier.complete_graph(graph, None)

            mod_ops, mod_extras = modifier.create_ops(
                steps_per_epoch, global_step, graph
            )

            with tf_compat.Session() as sess:
                sess.run(tf_compat.global_variables_initializer())
                modifier.initialize_session(sess)
                modifier.complete_graph(graph, sess)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
class ScheduledModifierTest(ModifierTest, BaseScheduledTest):
    # noinspection PyMethodOverriding
    def test_props_start(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props_start(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props_end(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props_end(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
class ScheduledUpdateModifierTest(ScheduledModifierTest, BaseUpdateTest):
    # noinspection PyMethodOverriding
    def test_props_frequency(
        self,
        modifier_lambda: Callable[[], ScheduledUpdateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props_frequency(modifier_lambda, framework=TENSORFLOW_V1_FRAMEWORK)


@TensorFlowModifierYAML()
class ModifierImpl(Modifier):
    def __init__(self):
        super().__init__()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize("modifier_lambda", [ModifierImpl], scope="function")
@pytest.mark.parametrize(
    "graph_lambda", [mlp_graph_lambda, conv_graph_lambda], scope="function"
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestModifierImpl(ModifierTest):
    pass


@TensorFlowModifierYAML()
class ScheduledModifierImpl(ScheduledModifier):
    def __init__(
        self,
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
    ):
        super().__init__()


@pytest.mark.parametrize("modifier_lambda", [ScheduledModifierImpl], scope="function")
@pytest.mark.parametrize(
    "graph_lambda", [mlp_graph_lambda, conv_graph_lambda], scope="function"
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestScheduledModifierImpl(ScheduledModifierTest):
    pass


@TensorFlowModifierYAML()
class ScheduledUpdateModifierImpl(ScheduledUpdateModifier):
    def __init__(
        self,
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
        update_frequency: float = -1,
    ):
        super().__init__()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "modifier_lambda", [ScheduledUpdateModifierImpl], scope="function"
)
@pytest.mark.parametrize(
    "graph_lambda", [mlp_graph_lambda, conv_graph_lambda], scope="function"
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestScheduledUpdateModifierImpl(ScheduledUpdateModifierTest):
    pass
