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


RECIPE_LEVEL_0 = """
init_sparsity: 0.2
final_sparsity: 0.8

modifiers:
    - !EpochRangeModifier
        end_epoch: 1.0
        start_epoch: 0.0

    - !GMPruningModifier
        init_sparsity: eval(init_sparsity)
        final_sparsity: eval(final_sparsity)
        start_epoch: 0.0
        end_epoch: 3.0
        update_frequency: 1.0
        params: ["re:.*weight"]
        leave_enabled: True
        inter_func: cubic
        log_types: __ALL__
        mask_type: unstructured
"""

RECIPE_LEVEL_1 = """
compression_sparsity = 0.5

modifiers:
    - !EpochRangeModifier
        end_epoch: 6.0
        start_epoch: 0.0
  
pruning_modifiers:
     - !ACDCPruningModifier
        compression_sparsity: eval(compression_sparsity)
        start_epoch: 2.0
        end_epoch: 6.0
        update_frequency: 2
        params: ['re:.*conv*', 're:.*fc.weight*']
"""


RECIPE_LEVEL_0_EVAL = """
version: 1.1.0

modifiers:
    - !EpochRangeModifier
        end_epoch: 1.0
        start_epoch: 0.0

    - !GMPruningModifier
        end_epoch: 3.0
        final_sparsity: 0.8
        init_sparsity: 0.2
        inter_func: cubic
        leave_enabled: True
        log_types: __ALL__
        mask_type: unstructured
        params: ['re:.*weight']
        start_epoch: 0.0
        update_frequency: 1.0

metadata: {'metadata': None, 'level': 0}
"""

RECIPE_LEVEL_1_EVAL = """
"""

METADATA_LEVEL_0 = {'metadata': None, 'level': 0}
METADATA_LEVEL_1 = {'metadata': None, 'level': 1}
import tempfile
import yaml
@pytest.mark.parametrize(
    "recipe,metadata,checkpoint_recipe,expected_recipe,raise_value_error",
    [
        (RECIPE_LEVEL_0,METADATA_LEVEL_0,None,RECIPE_LEVEL_0_EVAL, False),
        (RECIPE_LEVEL_1,METADATA_LEVEL_1,RECIPE_LEVEL_0_EVAL, RECIPE_LEVEL_1_EVAL, False),

    ],
)
def test_lifecycle_manager_staged(recipe, metadata, checkpoint_recipe, expected_recipe, raise_value_error):
    temp_dir = tempfile.mkdtemp()
    recipe_path = os.path.join(temp_dir, 'recipy.yaml')
    recipe_manager = ScheduledModifierManager.from_yaml(file_path=recipe, metadata=metadata)
    checkpoint_manager = None if checkpoint_recipe else None
    recipe_manager.save(recipe_path, checkpoint_manager)

    with open(recipe_path, 'r') as file:
        final_recipe = file.read()
    assert final_recipe == expected_recipe





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
