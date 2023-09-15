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
import platform
from copy import deepcopy

import pytest

from sparseml import version as sparseml_version
from sparseml.optim import (
    add_framework_metadata,
    check_if_staged_recipe,
    evaluate_recipe_yaml_str_equations,
    load_recipe_yaml_str,
    load_recipe_yaml_str_no_classes,
    update_recipe_variables,
    validate_metadata,
)
from sparseml.utils import FRAMEWORK_METADATA_KEY, RECIPE_METADATA_KEY


STAGED_RECIPE_COMPLEX = """
sparsity: {sparsity}
lr_func: {lr_func}
init_lr: 0.05
final_lr: 0.0
end_epoch: 100
update_frequency: 10
start_epoch: 0.0
global_sparsity: True

ac_dc_phase:
  update_frequency: 5
  end_epoch_global: eval(end_epoch)
  end_epoch: 75
  num_epochs: {num_epochs}
  end_warm_up_epoch: 5.0
  final_lr: eval(final_lr)
  final_lr: 0.256
  sparsity: 0.5
  global_sparsity: False

  training_modifiers:
  - !EpochRangeModifier
    start_epoch: eval(start_epoch)
    end_epoch: eval(end_epoch_global)

  - !LearningRateFunctionModifier
    start_epoch: eval(start_epoch)
    end_epoch: eval(end_warm_up_epoch)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(end_warm_up_epoch)
    end_epoch: eval(end_epoch_global)
    lr_func: eval(lr_func)
    init_lr: eval(final_lr)
    final_lr: 0.0

  pruning_modifiers:

  - !ACDCPruningModifier
     compression_sparsity: eval(sparsity)
     start_epoch: 1.0
     end_epoch: eval(end_epoch)
     update_frequency: eval(update_frequency)
     params: ['re:.*conv*', 're:.*fc.weight*']
     global_sparsity: eval(global_sparsity)

next_stage:
  new_num_epochs: {new_num_epochs}

  modifiers:
    - !EpochRangeModifier
        end_epoch: eval(end_epoch)
        start_epoch: eval(start_epoch + new_num_epochs)

    - !GMPruningModifier
        end_epoch: eval(end_epoch)
        final_sparsity: eval(sparsity)
        init_sparsity: eval(sparsity)
"""

STAGED_RECIPE_COMPLEX_EVAL = """
sparsity: 0.9
lr_func: cosine
init_lr: 0.05
final_lr: 0.0
end_epoch: 100
update_frequency: 10
start_epoch: 0.0
global_sparsity: True

ac_dc_phase:
  update_frequency: 5
  end_epoch_global: 100
  end_epoch: 75
  num_epochs: 10
  end_warm_up_epoch: 5.0
  final_lr: 0.256
  sparsity: 0.5
  global_sparsity: False

  training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 100

  - !LearningRateFunctionModifier
    start_epoch: 0.0
    end_epoch: 5.0
    lr_func: linear
    init_lr: 0.05
    final_lr: 0.256

  - !LearningRateFunctionModifier
    start_epoch: 5.0
    end_epoch: 100
    lr_func: cosine
    init_lr: 0.256
    final_lr: 0.0

  pruning_modifiers:

  - !ACDCPruningModifier
     compression_sparsity: 0.5
     start_epoch: 1.0
     end_epoch: 75
     update_frequency: 5
     params: ['re:.*conv*', 're:.*fc.weight*']
     global_sparsity: False

next_stage:
  new_num_epochs: 15

  modifiers:
    - !EpochRangeModifier
        end_epoch: 100
        start_epoch: 15.0

    - !GMPruningModifier
        end_epoch: 100
        final_sparsity: 0.9
        init_sparsity: 0.9
"""

STAGED_RECIPE_SIMPLE = """
first_variable: 10
second_variable: 5
lr_multiplier: 2

first_stage:
  lr: 0.1
  num_epochs: 10
  init_lr: eval(lr * 2)
  final_lr: eval(lr)

  training_modifiers:
    - !EpochRangeModifier
        end_epoch: eval(num_epochs + first_variable)
        start_epoch: 0.0

    - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: eval(num_epochs)
      lr_func: linear
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)

next_stage:
  new_num_epochs: 15
  sparsity: 0.9

  modifiers:
    - !EpochRangeModifier
        end_epoch: eval(new_num_epochs)
        start_epoch: eval(second_variable)

    - !GMPruningModifier
        end_epoch: eval(second_variable + first_variable)
        final_sparsity: eval(sparsity)
        init_sparsity: eval(sparsity)
"""

STAGED_RECIPE_SIMPLE_EVAL = """
first_variable: 10
second_variable: 5
lr_multiplier: 2

first_stage:
  lr: 0.1
  num_epochs: 10
  init_lr: 0.2
  final_lr: 0.1

  training_modifiers:
    - !EpochRangeModifier
        end_epoch: 20
        start_epoch: 0.0

    - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: 10
      lr_func: linear
      init_lr: 0.2
      final_lr: 0.1

next_stage:
  new_num_epochs: 15
  sparsity: 0.9

  modifiers:
    - !EpochRangeModifier
        end_epoch: 15
        start_epoch: 5

    - !GMPruningModifier
        end_epoch: 15
        final_sparsity: 0.9
        init_sparsity: 0.9
"""

RECIPE_SIMPLE_EVAL = """
num_epochs: 10.0
pruning_start_epoch: eval(num_epochs * 0.2)
pruning_end_epoch: eval(num_epochs * 0.8)
init_sparsity: 0.2
num_pruning_epochs: 6
pruning_mask_type: [1, 4]

modifiers:
    - !EpochRangeModifier
        end_epoch: 1.0
        start_epoch: 0.0

    - !GMPruningModifier
        end_epoch: eval(pruning_end_epoch)
        final_sparsity: eval(init_sparsity + 0.7)
        init_sparsity: eval(init_sparsity)
        inter_func: cubic
        leave_enabled: True
        mask_type: eval(pruning_mask_type)
        params: __ALL_PRUNABLE__
        start_epoch: eval(pruning_start_epoch)
        update_frequency: 0.01
"""

RECIPE_MULTI_EVAL = """
num_epochs: 10.0
pruning_end_epoch: eval(pruning_start_epoch + num_pruning_epochs)
pruning_start_epoch: eval(num_epochs * 0.2)
num_pruning_epochs: 6
init_sparsity: 0.2
pruning_mask_type: [1, 4]

modifiers:
    - !EpochRangeModifier
        end_epoch: 1.0
        start_epoch: 0.0

    - !GMPruningModifier
        end_epoch: eval(pruning_end_epoch)
        final_sparsity: eval(init_sparsity + 0.7)
        init_sparsity: eval(init_sparsity)
        inter_func: cubic
        leave_enabled: True
        mask_type: eval(pruning_mask_type)
        params: __ALL_PRUNABLE__
        start_epoch: eval(pruning_start_epoch)
        update_frequency: 0.01
"""

TARGET_RECIPE = """
num_epochs: {num_epochs}
init_sparsity: 0.2
pruning_start_epoch: 2.0
pruning_end_epoch: 8.0
num_pruning_epochs: 6
pruning_mask_type: [1, 4]

modifiers:
- !EpochRangeModifier
    end_epoch: 1.0
    start_epoch: 0.0
- !GMPruningModifier
    end_epoch: 8.0
    final_sparsity: 0.8999999999999999
    init_sparsity: 0.2
    inter_func: cubic
    leave_enabled: true
    mask_type:
    - 1
    - 4
    params: __ALL_PRUNABLE__
    start_epoch: 2.0
    update_frequency: 0.01
pruning_start_epoch: 2.0
"""

RECIPE_SIMPLE_EVAL_W_METADATA = """
num_epochs: 10.0
pruning_start_epoch: eval(num_epochs * 0.2)
pruning_end_epoch: eval(num_epochs * 0.8)
init_sparsity: 0.2
num_pruning_epochs: 6
__metadata__:
  this: is
  metadata: 90

modifiers:
    - !EpochRangeModifier
        end_epoch: 1.0
        start_epoch: 0.0

    - !GMPruningModifier
        end_epoch: eval(pruning_end_epoch)
        final_sparsity: eval(init_sparsity + 0.7)
        init_sparsity: eval(init_sparsity)
        inter_func: cubic
        leave_enabled: True
        log_types: __ALL__
        mask_type: [1, 4]
        params: __ALL_PRUNABLE__
        start_epoch: eval(pruning_start_epoch)
        update_frequency: 0.01
"""

STAGED_RECIPE_SIMPLE_EVAL_W_METADATA = """
first_variable: 10
second_variable: 5
lr_multiplier: 2

first_stage:
  __metadata__:
    this: is
    metadata: 110
  lr: 0.1
  num_epochs: 10
  init_lr: 0.2
  final_lr: 0.1

  training_modifiers:
    - !EpochRangeModifier
        end_epoch: 20
        start_epoch: 0.0

    - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: 10
      lr_func: linear
      init_lr: 0.2
      final_lr: 0.1

next_stage:
  __metadata__:
    this: is
    metadata: 120
  new_num_epochs: 15
  sparsity: 0.9

  modifiers:
    - !EpochRangeModifier
        end_epoch: 15
        start_epoch: 5

    - !GMPruningModifier
        end_epoch: 15
        final_sparsity: 0.9
        init_sparsity: 0.9
"""


def _test_nested_equality(val, other):
    assert type(val) == type(other)
    if isinstance(val, list):
        assert len(val) == len(other)
        for v, o in zip(val, other):
            _test_nested_equality(v, o)
    elif isinstance(val, dict):
        assert len(val) == len(other)
        for key in val:
            assert key in other
            _test_nested_equality(val[key], other[key])
    else:
        assert val == other


@pytest.mark.parametrize(
    "recipe,expected_recipe, is_staged",
    [
        (
            TARGET_RECIPE.format(num_epochs=10.0),
            TARGET_RECIPE.format(num_epochs=10.0),
            False,
        ),
        (RECIPE_SIMPLE_EVAL, TARGET_RECIPE.format(num_epochs=10.0), False),
        (RECIPE_MULTI_EVAL, TARGET_RECIPE.format(num_epochs=10.0), False),
        (STAGED_RECIPE_SIMPLE, STAGED_RECIPE_SIMPLE_EVAL, True),
        (
            STAGED_RECIPE_COMPLEX.format(
                sparsity=0.9, num_epochs=10, lr_func="cosine", new_num_epochs=15
            ),
            STAGED_RECIPE_COMPLEX_EVAL,
            True,
        ),
    ],
)
def test_evaluate_recipe_yaml_str_equations(recipe, expected_recipe, is_staged):
    evaluated_recipe = evaluate_recipe_yaml_str_equations(recipe)
    evaluated_yaml = load_recipe_yaml_str_no_classes(evaluated_recipe)
    expected_is_staged = check_if_staged_recipe(evaluated_yaml)
    expected_yaml = load_recipe_yaml_str_no_classes(expected_recipe)
    assert expected_is_staged == is_staged
    assert isinstance(evaluated_yaml, dict)
    assert isinstance(expected_yaml, dict)
    _test_nested_equality(evaluated_yaml, expected_yaml)


def _generate_fake_metadata(item1=("this", "is"), item2=("metadata", 100)):
    return {k: v for (k, v) in (item1, item2)}


@pytest.mark.parametrize(
    "metadata,yaml_str, expected_metadata, raise_warning",
    [
        # Testing simple recipe
        (
            _generate_fake_metadata(),
            RECIPE_SIMPLE_EVAL,
            {RECIPE_METADATA_KEY: _generate_fake_metadata()},
            False,
        ),
        # Testing simple recipe (metadata = None)
        (None, RECIPE_SIMPLE_EVAL, {RECIPE_METADATA_KEY: None}, False),
        # Testing simple recipe, attempting to overwrite previous metadata
        (
            _generate_fake_metadata(item2=("metadata", 120)),
            RECIPE_SIMPLE_EVAL_W_METADATA,
            {RECIPE_METADATA_KEY: _generate_fake_metadata(item2=("metadata", 120))},
            True,
        ),
        # Testing simple recipe, previous metadata present but new metadata is None
        (
            None,
            RECIPE_SIMPLE_EVAL_W_METADATA,
            {RECIPE_METADATA_KEY: _generate_fake_metadata(item2=("metadata", 90))},
            False,
        ),
        # Testing simple recipe, passing new metadata
        # which is equal to previous metadata.
        (
            _generate_fake_metadata(item2=("metadata", 90)),
            RECIPE_SIMPLE_EVAL_W_METADATA,
            {RECIPE_METADATA_KEY: _generate_fake_metadata(item2=("metadata", 90))},
            False,
        ),
        # Testing staged recipe
        (
            _generate_fake_metadata(item2=("metadata", 150)),
            STAGED_RECIPE_SIMPLE_EVAL,
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 150)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 150)),
            },
            False,
        ),
        # Testing staged recipe (metadata = None)
        (
            None,
            STAGED_RECIPE_SIMPLE_EVAL,
            {"first_stage": None, "next_stage": None},
            False,
        ),
        # Testing staged recipe, attempting to overwrite previous metadata
        (
            _generate_fake_metadata(item2=("metadata", 150)),
            STAGED_RECIPE_SIMPLE_EVAL_W_METADATA,
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 150)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 150)),
            },
            True,
        ),
        # Testing staged recipe, previous metadata present but new metadata is None
        (
            None,
            STAGED_RECIPE_SIMPLE_EVAL_W_METADATA,
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 110)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 120)),
            },
            False,
        ),
        # Testing staged recipe, passing new staged metadata
        # which is equal to previous metadata.
        (
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 110)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 120)),
            },
            STAGED_RECIPE_SIMPLE_EVAL_W_METADATA,
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 110)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 120)),
            },
            False,
        ),
        # Testing staged recipe, passing new staged metadata
        # which is equal to previous metadata.
        (
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 160)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 140)),
            },
            STAGED_RECIPE_SIMPLE_EVAL_W_METADATA,
            {
                "first_stage": _generate_fake_metadata(item2=("metadata", 160)),
                "next_stage": _generate_fake_metadata(item2=("metadata", 140)),
            },
            True,
        ),
    ],
)
def test_validate_metadata(
    metadata, yaml_str, expected_metadata, raise_warning, caplog
):
    with caplog.at_level(logging.WARNING):
        metadata = validate_metadata(metadata, yaml_str)
        assert raise_warning == bool(caplog.text)
    assert metadata == expected_metadata


metadata_w_framework_1 = _generate_fake_metadata()
metadata_w_framework_1[FRAMEWORK_METADATA_KEY] = {
    "python_version": platform.python_version(),
    "sparseml_version": sparseml_version,
}
metadata_w_framework_2 = deepcopy(metadata_w_framework_1)
metadata_w_framework_2[FRAMEWORK_METADATA_KEY]["python_version"] = "placeholder"


@pytest.mark.parametrize(
    "metadata, expected_metadata, raise_warning",
    [
        # pass unstaged metadata without framework metadata
        (
            {RECIPE_METADATA_KEY: _generate_fake_metadata()},
            {RECIPE_METADATA_KEY: metadata_w_framework_1},
            False,
        ),
        # pass staged metadata without framework metadata
        (
            {
                "stage_0": _generate_fake_metadata(),
                "stage_1": _generate_fake_metadata(),
            },
            {"stage_0": metadata_w_framework_1, "stage_1": metadata_w_framework_1},
            False,
        ),
        # pass unstaged metadata with framework metadata
        (
            {RECIPE_METADATA_KEY: metadata_w_framework_2},
            {RECIPE_METADATA_KEY: metadata_w_framework_1},
            True,
        ),
        # pass staged metadata with framework metadata
        (
            {"stage_0": metadata_w_framework_2, "stage_1": metadata_w_framework_2},
            {"stage_0": metadata_w_framework_1, "stage_1": metadata_w_framework_1},
            True,
        ),
    ],
)
def test_add_framework_metadata(metadata, expected_metadata, raise_warning, caplog):
    with caplog.at_level(logging.WARNING):
        metadata = add_framework_metadata(metadata)
        assert raise_warning == bool(caplog.text)

    assert metadata == expected_metadata


RECIPE_INVALID_LOOP = """
val_1: eval(val_2)
val_2: eval(val_1)
"""

RECIPE_INVALID_UNDEFINED = """
val_1: eval(val2)
"""

RECIPE_INVALID_DTYPE = """
val_1: eval([1,2])
"""


@pytest.mark.parametrize(
    "recipe",
    [
        RECIPE_INVALID_LOOP,
        RECIPE_INVALID_UNDEFINED,
        RECIPE_INVALID_DTYPE,
    ],
)
def test_evaluate_recipe_yaml_str_equations_invalid(recipe):
    with pytest.raises(RuntimeError):
        evaluate_recipe_yaml_str_equations(recipe)


@pytest.mark.parametrize(
    "zoo_path",
    [
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/"
            "pruned-conservative"
        ),
        ("zoo:bert-base-wikipedia_bookcorpus-pruned90?recipe=transfer_question"),
    ],
)
def test_load_recipe_yaml_str_zoo(zoo_path):
    assert load_recipe_yaml_str(zoo_path)


@pytest.mark.parametrize(
    "base_recipe,override_variables,target_recipe,raises_value_error",
    [
        (
            TARGET_RECIPE.format(num_epochs=100.0),
            {"num_epochs": 10.0},
            TARGET_RECIPE.format(num_epochs=10.0),
            False,
        ),
        (
            TARGET_RECIPE.format(num_epochs=100.0),
            {"invalid_var_name": 10.0},
            TARGET_RECIPE.format(num_epochs=10.0),
            True,
        ),
        (
            STAGED_RECIPE_COMPLEX.format(
                sparsity=0.9, num_epochs=10, lr_func="linear", new_num_epochs=15
            ),
            {
                "sparsity": 0.5,
                "num_epochs": 11,
                "lr_func": "cosine",
                "new_num_epochs": 13,
            },
            STAGED_RECIPE_COMPLEX.format(
                sparsity=0.5, num_epochs=11, lr_func="cosine", new_num_epochs=13
            ),
            False,
        ),
        (
            STAGED_RECIPE_COMPLEX.format(
                sparsity=0.9, num_epochs=10, lr_func="cosine", new_num_epochs=15
            ),
            {"invalid_var_name": 10.0},
            STAGED_RECIPE_COMPLEX.format(
                sparsity=0.5, num_epochs=11, lr_func="linear", new_num_epochs=13
            ),
            True,
        ),
    ],
)
def test_update_recipe_variables(
    base_recipe, override_variables, target_recipe, raises_value_error
):
    if raises_value_error:
        with pytest.raises(ValueError):
            update_recipe_variables(base_recipe, override_variables)

    else:
        updated_recipe = update_recipe_variables(base_recipe, override_variables)
        updated_yaml = load_recipe_yaml_str_no_classes(updated_recipe)
        target_yaml = load_recipe_yaml_str_no_classes(target_recipe)
        _test_nested_equality(updated_yaml, target_yaml)
