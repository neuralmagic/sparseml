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

import pytest

from sparseml.optim import (
    evaluate_recipe_yaml_str_equations,
    load_recipe_yaml_str,
    load_recipe_yaml_str_no_classes,
    update_recipe_variables,
)


RECIPE_SIMPLE_EVAL = """
num_epochs: 10.0
pruning_start_epoch: eval(num_epochs * 0.2)
pruning_end_epoch: eval(num_epochs * 0.8)
init_sparsity: 0.2
num_pruning_epochs: 6

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


RECIPE_MULTI_EVAL = """
num_epochs: 10.0
pruning_end_epoch: eval(pruning_start_epoch + num_pruning_epochs)
pruning_start_epoch: eval(num_epochs * 0.2)
num_pruning_epochs: 6
init_sparsity: 0.2

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

TARGET_RECIPE = """
num_epochs: {num_epochs}
init_sparsity: 0.2
pruning_start_epoch: 2.0
pruning_end_epoch: 8.0
num_pruning_epochs: 6

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
    log_types: __ALL__
    mask_type:
    - 1
    - 4
    params: __ALL_PRUNABLE__
    start_epoch: 2.0
    update_frequency: 0.01
pruning_start_epoch: 2.0
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
    "recipe,expected_recipe",
    [
        (TARGET_RECIPE.format(num_epochs=10.0), TARGET_RECIPE.format(num_epochs=10.0)),
        (RECIPE_SIMPLE_EVAL, TARGET_RECIPE.format(num_epochs=10.0)),
        (RECIPE_MULTI_EVAL, TARGET_RECIPE.format(num_epochs=10.0)),
    ],
)
def test_evaluate_recipe_yaml_str_equations(recipe, expected_recipe):
    evaluated_recipe = evaluate_recipe_yaml_str_equations(recipe)
    evaluated_yaml = load_recipe_yaml_str_no_classes(evaluated_recipe)
    expected_yaml = load_recipe_yaml_str_no_classes(expected_recipe)

    assert isinstance(evaluated_yaml, dict)
    assert isinstance(expected_yaml, dict)
    _test_nested_equality(evaluated_yaml, expected_yaml)


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
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/"
            "pruned-conservative?recipe_type=original"
        ),
    ],
)
def test_load_recipe_yaml_str_zoo(zoo_path):
    assert load_recipe_yaml_str(zoo_path)


@pytest.mark.parametrize(
    "base_recipe,override_variables,target_recipe",
    [
        (
            TARGET_RECIPE.format(num_epochs=100.0),
            {"num_epochs": 10.0},
            TARGET_RECIPE.format(num_epochs=10.0),
        ),
    ],
)
def test_update_recipe_variables(base_recipe, override_variables, target_recipe):
    updated_recipe = update_recipe_variables(base_recipe, override_variables)

    updated_yaml = load_recipe_yaml_str_no_classes(updated_recipe)
    target_yaml = load_recipe_yaml_str_no_classes(target_recipe)
    _test_nested_equality(updated_yaml, target_yaml)
