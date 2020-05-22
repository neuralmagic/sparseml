import pytest

import os

from typing import Callable, Dict, List, Union

from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.recal import (
    TENSORFLOW_FRAMEWORK,
    TensorFlowModifierYAML,
    Modifier,
    ScheduledModifier,
    ScheduledUpdateModifier,
)

from tests.recal.test_modifier import (
    BaseModifierTest,
    BaseScheduledTest,
    BaseUpdateTest,
)
from tests.tensorflow.helpers import mlp_net, conv_net


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
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
class ModifierTest(BaseModifierTest):
    # noinspection PyMethodOverriding
    def test_constructor(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_constructor(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_yaml(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml_key(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_yaml_key(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_repr(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_repr(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)

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
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
class ScheduledModifierTest(ModifierTest, BaseScheduledTest):
    # noinspection PyMethodOverriding
    def test_props_start(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props_start(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props_end(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props_end(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
class ScheduledUpdateModifierTest(ScheduledModifierTest, BaseUpdateTest):
    # noinspection PyMethodOverriding
    def test_props_frequency(
        self,
        modifier_lambda: Callable[[], ScheduledUpdateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        super().test_props_frequency(modifier_lambda, framework=TENSORFLOW_FRAMEWORK)


@TensorFlowModifierYAML()
class ModifierImpl(Modifier):
    def __init__(self, log_types: Union[str, List[str]] = ["python"]):
        super().__init__(log_types)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
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
        log_types: Union[str, List[str]] = ["python"],
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
    ):
        super().__init__(log_types)


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
        log_types: Union[str, List[str]] = ["python"],
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
        update_frequency: float = -1,
    ):
        super().__init__(log_types)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
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
