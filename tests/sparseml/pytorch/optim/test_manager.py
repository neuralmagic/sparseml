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
from collections import OrderedDict
from typing import Callable

import pytest
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier
from sparseml.pytorch.optim import Modifier, ScheduledModifierManager
from tests.sparseml.pytorch.helpers import (
    SAMPLE_STAGED_RECIPE,
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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_manager_yaml():
    manager = ScheduledModifierManager([ScheduledModifierImpl()])
    yaml_str = str(manager)
    assert yaml_str


@pytest.mark.parametrize("staged_recipe", [SAMPLE_STAGED_RECIPE])
def test_manager_staged_recipe_serialization(staged_recipe):
    manager = ScheduledModifierManager.from_yaml(staged_recipe)
    assert isinstance(manager.modifiers, OrderedDict)

    manager_yaml_str = str(manager)
    reloaded_manager = ScheduledModifierManager.from_yaml(manager_yaml_str)
    isinstance(reloaded_manager.modifiers, OrderedDict)

    # test iter modifiers
    modifiers_list = list(manager.iter_modifiers())
    reloaded_modifiers_list = list(reloaded_manager.iter_modifiers())
    assert len(modifiers_list) == len(reloaded_modifiers_list) > 0
    for mod, reloaded_mod in zip(modifiers_list, reloaded_modifiers_list):
        assert isinstance(mod, BaseModifier)
        assert type(mod) is type(reloaded_mod)

    # test stages dict
    assert len(manager.modifiers) == len(reloaded_manager.modifiers)
    for stage_name, reloaded_stage_name in zip(
        manager.modifiers, reloaded_manager.modifiers
    ):
        assert stage_name == reloaded_stage_name
        stage_modifiers = manager.modifiers[stage_name]
        reloaded_stage_modifiers = reloaded_manager.modifiers[reloaded_stage_name]
        assert isinstance(stage_modifiers, list)
        assert isinstance(reloaded_stage_modifiers, list)
        assert len(stage_modifiers) == len(reloaded_stage_modifiers) > 0
        assert [type(mod) for mod in stage_modifiers] == (
            [type(mod) for mod in reloaded_stage_modifiers]
        )
