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

import logging
import os
import platform
from collections import OrderedDict
from typing import Callable

import pytest
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml import version as sparseml_version
from sparseml.optim import BaseModifier
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.sparsification import (
    LayerThinningModifier,
    LearningRateFunctionModifier,
    Modifier,
)
from tests.sparseml.pytorch.helpers import (
    SAMPLE_STAGED_RECIPE,
    LinearNet,
    create_optim_adam,
    create_optim_sgd,
)
from tests.sparseml.pytorch.sparsification.test_modifier import (
    ModifierTest,
    ScheduledModifierImpl,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


STANDARD_RECIPE_1 = """
init_sparsity: 0.2
final_sparsity: 0.8

modifiers:
    - !EpochRangeModifier
        end_epoch: 3.0
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
        mask_type: unstructured
"""

STANDARD_RECIPE_2 = """
params: ["re:.*weight"]

training_modifiers:
    - !EpochRangeModifier
        end_epoch: 6.0
        start_epoch: 0.0

pruning_modifiers:
    - !ConstantPruningModifier
        start_epoch: 2.0
        end_epoch: 5.0
        params: eval(params)
"""


STANDARD_RECIPE_1_EVAL = """version: 1.1.0

__metadata__:
  metadata: None
  level: 0
  framework_metadata:
    python_version: {python_version}
    sparseml_version: {sparseml_version}
    torch_version: {torch_version}

modifiers:
    - !EpochRangeModifier
        end_epoch: 3.0
        start_epoch: 0.0

    - !GMPruningModifier
        end_epoch: 3.0
        final_sparsity: 0.8
        global_sparsity: False
        init_sparsity: 0.2
        inter_func: cubic
        leave_enabled: True
        mask_type: unstructured
        params: ['re:.*weight']
        start_epoch: 0.0
        update_frequency: 1.0
"""

TWO_STAGES_RECIPE = """version: 1.1.0

{stage_0_name}:
  __metadata__:
    metadata: None
    level: 0
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  {stage_0_name}_modifiers:
      - !EpochRangeModifier
          end_epoch: 3.0
          start_epoch: 0.0
  
      - !GMPruningModifier
          end_epoch: 3.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 0.0
          update_frequency: 1.0
  

{stage_1_name}:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  {stage_1_name}_modifiers:
      - !ConstantPruningModifier
          end_epoch: 8.0
          params: ['re:.*weight']
          start_epoch: 5.0
          update_frequency: -1
  
      - !EpochRangeModifier
          end_epoch: 9.0
          start_epoch: 3.0
  
"""  # noqa: W293


THREE_STAGES_RECIPE_1 = """version: 1.1.0

stage_0:
  __metadata__:
    metadata: None
    level: 0
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_0_modifiers:
      - !EpochRangeModifier
          end_epoch: 3.0
          start_epoch: 0.0
  
      - !GMPruningModifier
          end_epoch: 3.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 0.0
          update_frequency: 1.0
  

stage_1:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_1_modifiers:
      - !ConstantPruningModifier
          end_epoch: 8.0
          params: ['re:.*weight']
          start_epoch: 5.0
          update_frequency: -1
  
      - !EpochRangeModifier
          end_epoch: 9.0
          start_epoch: 3.0
  

stage_3:
  __metadata__:
    metadata: None
    level: 3
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_3_modifiers:
      - !EpochRangeModifier
          end_epoch: 12.0
          start_epoch: 9.0
  
      - !GMPruningModifier
          end_epoch: 12.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 9.0
          update_frequency: 1.0
  
"""  # noqa: W293
THREE_STAGES_RECIPE_2 = """version: 1.1.0

pre_stage_0:
  __metadata__:
    metadata: None
    level: 0
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  pre_stage_0_modifiers:
      - !EpochRangeModifier
          end_epoch: 3.0
          start_epoch: 0.0
  
      - !GMPruningModifier
          end_epoch: 3.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 0.0
          update_frequency: 1.0
  

stage_0:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_0_modifiers:
      - !EpochRangeModifier
          end_epoch: 6.0
          start_epoch: 3.0
  
      - !GMPruningModifier
          end_epoch: 6.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 3.0
          update_frequency: 1.0
  

stage_1:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_1_modifiers:
      - !ConstantPruningModifier
          end_epoch: 11.0
          params: ['re:.*weight']
          start_epoch: 8.0
          update_frequency: -1
  
      - !EpochRangeModifier
          end_epoch: 12.0
          start_epoch: 6.0
  
"""  # noqa: W293

FOUR_STAGES_RECIPE = """version: 1.1.0

stage_0:
  __metadata__:
    metadata: None
    level: 0
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_0_modifiers:
      - !EpochRangeModifier
          end_epoch: 3.0
          start_epoch: 0.0
  
      - !GMPruningModifier
          end_epoch: 3.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 0.0
          update_frequency: 1.0
  

stage_1:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_1_modifiers:
      - !ConstantPruningModifier
          end_epoch: 8.0
          params: ['re:.*weight']
          start_epoch: 5.0
          update_frequency: -1
  
      - !EpochRangeModifier
          end_epoch: 9.0
          start_epoch: 3.0
  

stage_3:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_3_modifiers:
      - !EpochRangeModifier
          end_epoch: 12.0
          start_epoch: 9.0
  
      - !GMPruningModifier
          end_epoch: 12.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 9.0
          update_frequency: 1.0
  

stage_4:
  __metadata__:
    metadata: None
    level: 1
    framework_metadata:
      python_version: {python_version}
      sparseml_version: {sparseml_version}
      torch_version: {torch_version}

  stage_4_modifiers:
      - !ConstantPruningModifier
          end_epoch: 17.0
          params: ['re:.*weight']
          start_epoch: 14.0
          update_frequency: -1
  
      - !EpochRangeModifier
          end_epoch: 18.0
          start_epoch: 12.0
  
"""  # noqa: W293

RECIPE_END_EPOCH_IMPLICIT = """
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 52.0

  - !SetLearningRateModifier
    start_epoch: 50.0
    learning_rate: 0.000002

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: 50.0
    disable_quantization_observer_epoch: 51.0
    freeze_bn_stats_epoch: 51.0
    submodules: ['model.0']
"""

COMPOSED_RECIPE_END_EPOCH_IMPLICIT = """version: 1.1.0

stage_0:
  __metadata__: None

  stage_0_modifiers:
      - !ConstantPruningModifier
          end_epoch: 52
          params: __ALL_PRUNABLE__
          start_epoch: 0.0
          update_frequency: -1
  
      - !EpochRangeModifier
          end_epoch: 52.0
          start_epoch: 0.0
  
      - !QuantizationModifier
          activation_bits: 8
          disable_quantization_observer_epoch: 51.0
          end_epoch: 52
          exclude_batchnorm: True
          freeze_bn_stats_epoch: 51.0
          model_fuse_fn_name: conv_bn_relus
          quantize_conv_activations: True
          quantize_embedding_activations: True
          quantize_embeddings: True
          quantize_linear_activations: True
          reduce_range: False
          start_epoch: 50.0
          submodules: ['model.0']
          tensorrt: False
          weight_bits: 8
  
      - !SetLearningRateModifier
          constant_logging: False
          end_epoch: 52
          learning_rate: 2e-06
          start_epoch: 50.0
  

stage_1:
  __metadata__: None

  stage_1_modifiers:
      - !EpochRangeModifier
          end_epoch: 104.0
          start_epoch: 52.0
  
      - !ConstantPruningModifier
          end_epoch: -1.0
          params: __ALL_PRUNABLE__
          start_epoch: 52.0
          update_frequency: -1
  
      - !QuantizationModifier
          activation_bits: 8
          disable_quantization_observer_epoch: 103.0
          end_epoch: -1.0
          exclude_batchnorm: True
          freeze_bn_stats_epoch: 103.0
          model_fuse_fn_name: conv_bn_relus
          quantize_conv_activations: True
          quantize_embedding_activations: True
          quantize_embeddings: True
          quantize_linear_activations: True
          reduce_range: False
          start_epoch: 102.0
          submodules: ['model.0']
          tensorrt: False
          weight_bits: 8
  
      - !SetLearningRateModifier
          constant_logging: False
          end_epoch: -1.0
          learning_rate: 2e-06
          start_epoch: 102.0
  
"""  # noqa: W293


def _generate_fake_metadata(item1=("metadata", None), item2=("level", 1)):
    return {k: v for (k, v) in (item1, item2)}


python_version = platform.python_version()
torch_version = torch.__version__


@pytest.mark.parametrize(
    "recipe,checkpoint_recipe,metadata,expected_recipe,"
    "raise_warning, raise_value_error",
    [
        # Testing saving standard recipe with metadata, no stage composing
        (
            STANDARD_RECIPE_1,
            None,
            _generate_fake_metadata(item2=("level", 0)),
            STANDARD_RECIPE_1_EVAL.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            False,
            False,
        ),
        # Testing composing standard recipe (with metadata)
        # with a standard checkpoint recipe
        (
            STANDARD_RECIPE_2,
            STANDARD_RECIPE_1_EVAL.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            _generate_fake_metadata(),
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_0",
                stage_1_name="stage_1",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            False,
            False,
        ),
        # Testing composing standard recipe (with metadata)
        # with a staged checkpoint recipe
        (
            STANDARD_RECIPE_1,
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_0",
                stage_1_name="stage_1",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            _generate_fake_metadata(item2=("level", 3)),
            THREE_STAGES_RECIPE_1.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            False,
            False,
        ),
        # Testing composing staged recipe (with metadata)
        # with standard checkpoint recipe (with metadata)
        (
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_0",
                stage_1_name="stage_1",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            STANDARD_RECIPE_1_EVAL.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            _generate_fake_metadata(),
            THREE_STAGES_RECIPE_2.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            True,
            False,
        ),
        # Testing composing two staged recipes
        (
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_3",
                stage_1_name="stage_4",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_0",
                stage_1_name="stage_1",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            _generate_fake_metadata(),
            FOUR_STAGES_RECIPE.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            True,
            False,
        ),
        # Testing composing two stage recipes with
        # same stage names -> should raise ValueError
        (
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_0",
                stage_1_name="stage_1",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            TWO_STAGES_RECIPE.format(
                stage_0_name="stage_0",
                stage_1_name="stage_1",
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            None,
            FOUR_STAGES_RECIPE.format(
                python_version=python_version,
                sparseml_version=sparseml_version,
                torch_version=torch_version,
            ),
            False,
            True,
        ),
        # Testing composing two recipes with modifiers containing
        # implicit `end_epoch` attribution (i.e. `end_epoch = -1`)
        (
            RECIPE_END_EPOCH_IMPLICIT,
            RECIPE_END_EPOCH_IMPLICIT,
            None,
            COMPOSED_RECIPE_END_EPOCH_IMPLICIT,
            False,
            False,
        ),
    ],
)
def test_lifecycle_manager_staged(
    recipe,
    metadata,
    checkpoint_recipe,
    expected_recipe,
    raise_warning,
    raise_value_error,
    caplog,
):

    with caplog.at_level(logging.WARNING):
        recipe_manager = ScheduledModifierManager.from_yaml(
            file_path=recipe, metadata=metadata
        )
        assert raise_warning == bool(caplog.text)

    if checkpoint_recipe:
        if raise_value_error:
            with pytest.raises(ValueError):
                ScheduledModifierManager.compose_staged(
                    checkpoint_recipe, recipe_manager
                )
            return
        else:
            recipe_manager = ScheduledModifierManager.compose_staged(
                base_recipe=checkpoint_recipe,
                additional_recipe=recipe_manager,
            )
    final_recipe = str(recipe_manager)
    assert final_recipe == expected_recipe


THINNING_TEST_INPUTS = [
    (
        # Base recipe
        lambda: LearningRateFunctionModifier(
            lr_func="linear",
            init_lr=0.1,
            final_lr=0.001,
            start_epoch=0,
            end_epoch=10,
            update_frequency=0.5,
        ),
        # Additional recipe to start after the base one
        lambda: LayerThinningModifier(
            param_group_dependency_map={},
            start_epoch=0.0,
            update_epochs=[1.0, 2.0, 3.0],
        ),
        10,  # expected start_epoch
        [11.0, 12.0, 13.0],  # expected update_epochs
    )
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("test_inputs", THINNING_TEST_INPUTS, scope="function")
def test_composing_thinning_modifier(test_inputs):
    recipe_0 = test_inputs[0]()
    recipe_1 = test_inputs[1]()
    expected_start_epoch = test_inputs[2]
    expected_update_epochs = test_inputs[3]

    recipe_0_manager = ScheduledModifierManager([recipe_0])
    recipe_1_manager = ScheduledModifierManager([recipe_1])
    manager = ScheduledModifierManager.compose_staged(
        base_recipe=recipe_0_manager,
        additional_recipe=recipe_1_manager,
    )
    assert manager.num_stages() == 2
    thin_mod = manager.modifiers["stage_1"][0]
    assert type(thin_mod) == LayerThinningModifier
    assert thin_mod.start_epoch == expected_start_epoch
    assert thin_mod.update_epochs == expected_update_epochs


TWO_STAGES_RECIPE = """version: 1.1.0

stage_0:

  stage_0_modifiers:
      - !EpochRangeModifier
          end_epoch: 3.0
          start_epoch: 0.0

      - !GMPruningModifier
          end_epoch: 3.0
          final_sparsity: 0.8
          global_sparsity: False
          init_sparsity: 0.2
          inter_func: cubic
          leave_enabled: True
          mask_type: unstructured
          params: ['re:.*weight']
          start_epoch: 0.0
          update_frequency: 1.0


stage_1:

  stage_1_modifiers:
      - !ConstantPruningModifier
          end_epoch: 8.0
          params: ['re:.*weight']
          start_epoch: 5.0
          update_frequency: -1

      - !EpochRangeModifier
          end_epoch: 9.0
          start_epoch: 3.0

"""


@pytest.mark.parametrize(
    "recipe",
    [
        STANDARD_RECIPE_1,  # passing standard recipe without metadata
        TWO_STAGES_RECIPE,  # passing staged recipe without metadata
    ],
)
def test_lifecycle_manager_staged_no_metadata(recipe):
    recipe_manager = ScheduledModifierManager.from_yaml(file_path=recipe)
    recipe_old = str(recipe_manager)

    recipe_manager = ScheduledModifierManager.from_yaml(file_path=recipe_old)
    recipe = str(recipe_manager)

    assert recipe_old == recipe


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
