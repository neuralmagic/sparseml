import pytest

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
from tests.tensorflow.helpers import mlp_net


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

        graph, ops = modifier.create_ops(graph, steps_per_epoch, global_step)

        assert graph
        assert isinstance(graph, tf_compat.Graph)
        assert ops is not None
        assert isinstance(ops, List)
        assert modifier.initialized

    def test_create_extras(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

        graph, extras = modifier.create_extras(graph, steps_per_epoch, global_step)

        assert graph
        assert isinstance(graph, tf_compat.Graph)
        assert extras is not None
        assert isinstance(extras, Dict)
        assert modifier.initialized

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

        graph, ops = modifier.create_ops(graph, steps_per_epoch, global_step)
        graph, extras = modifier.create_ops(graph, steps_per_epoch, global_step)
        graph = modifier.complete_graph(graph)

        assert graph
        assert isinstance(graph, tf_compat.Graph)


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


@pytest.mark.parametrize("modifier_lambda", [ModifierImpl], scope="function")
@pytest.mark.parametrize("graph_lambda", [mlp_net], scope="function")
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
@pytest.mark.parametrize("graph_lambda", [mlp_net], scope="function")
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


@pytest.mark.parametrize(
    "modifier_lambda", [ScheduledUpdateModifierImpl], scope="function"
)
@pytest.mark.parametrize("graph_lambda", [mlp_net], scope="function")
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestScheduledUpdateModifierImpl(ScheduledUpdateModifierTest):
    pass
