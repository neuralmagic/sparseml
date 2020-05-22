import pytest

import os

from typing import Callable

from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.recal import ScheduledModifierManager, Modifier

from tests.tensorflow.recal.test_modifier import (
    ModifierTest,
    ScheduledModifierImpl,
    mlp_graph_lambda,
    conv_graph_lambda,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [lambda: ScheduledModifierManager([ScheduledModifierImpl()])],
    scope="function",
)
@pytest.mark.parametrize(
    "graph_lambda", [mlp_graph_lambda, conv_graph_lambda], scope="function"
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestManagerImpl(ModifierTest):
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        # no yaml tests for manager
        return


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False), reason="Skipping tensorflow tests",
)
def test_manager_yaml():
    manager = ScheduledModifierManager([ScheduledModifierImpl()])
    yaml_str = ScheduledModifierManager.list_to_yaml(manager.modifiers)
    assert yaml_str
