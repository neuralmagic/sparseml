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

import pytest
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.optim import Modifier, ScheduledModifierManager
from tests.sparseml.pytorch.helpers import (
    LinearNet,
    create_optim_adam,
    create_optim_sgd,
)
from tests.sparseml.pytorch.optim.test_modifier import (
    ModifierTest,
    ScheduledModifierImpl,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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
    def test_initialize(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_steps_per_epoch: float,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        self.initialize_helper(
            modifier, model, optimizer, steps_per_epoch=test_steps_per_epoch
        )
        assert modifier.initialized
        assert optimizer.step._with_modifiers  # assert modifier steps added

    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        # no yaml tests for manager
        return

    def test_lifecycle(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        manager = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        original_step_func = optimizer.step.__func__

        self.initialize_helper(
            manager, model, optimizer, steps_per_epoch=test_steps_per_epoch
        )
        assert manager.initialized
        assert optimizer.step._with_modifiers  # assert modifier steps added
        assert optimizer.step._original_step_func is original_step_func
        assert optimizer.step != original_step_func

        assert optimizer._steps == 0
        assert optimizer._epoch == 0.0
        for i in range(1, test_steps_per_epoch + 2):
            optimizer.step()
            assert optimizer._steps == i
        assert optimizer._epoch >= 1.0

        # test original step func is restored
        manager.finalize(model, optimizer)
        assert optimizer.step.__func__ is original_step_func


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_manager_yaml():
    manager = ScheduledModifierManager([ScheduledModifierImpl()])
    yaml_str = str(manager)
    assert yaml_str
