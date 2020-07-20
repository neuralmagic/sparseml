import os
from typing import Callable

import pytest
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from neuralmagicML.pytorch.recal import (
    ScheduledModifierManager,
    ScheduledModifier,
    PyTorchModifierYAML,
    Modifier,
)


from tests.pytorch.helpers import (
    test_epoch,
    test_steps_per_epoch,
    test_loss,
    LinearNet,
    create_optim_sgd,
    create_optim_adam,
)
from tests.pytorch.recal.test_modifier import ModifierTest, ScheduledModifierImpl


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [lambda: ScheduledModifierManager([ScheduledModifierImpl()])],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function"
)
class TestManagerImpl(ModifierTest):
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,
        test_steps_per_epoch: float,
    ):
        # no yaml tests for manager
        return


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_manager_yaml():
    manager = ScheduledModifierManager([ScheduledModifierImpl()])
    yaml_str = str(manager)
    assert yaml_str
