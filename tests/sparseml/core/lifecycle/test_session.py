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

import sparseml.core.session as sml
from sparseml.core.framework import Framework


def recipe_with_layer_prefix():
    layer_prefix = "decoder"
    recipe = f"""
    metadata:
        target_model:
            layer_prefix: {layer_prefix}
            architecture: "opt"

    test_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                targets: __ALL_PRUNABLE__
                start: 0
                end: 5
    """
    return recipe, layer_prefix


def recipe_without_layer_prefix():
    recipe = """
    test_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                targets: __ALL_PRUNABLE__
                start: 0
                end: 5
    """
    return recipe, None


@pytest.fixture
def model():
    # identity model
    return lambda x: x


@pytest.mark.parametrize(
    "recipe, expected_layer_prefix",
    [
        recipe_without_layer_prefix(),
        recipe_with_layer_prefix(),
    ],
)
def test_session_initialize_propagates_layer_prefix_to_model(
    recipe, expected_layer_prefix, model
):
    session = sml.active_session()
    session.initialize(framework=Framework.general, model=model, recipe=recipe)
    print(f"{session.state.model.layer_prefix=}, {expected_layer_prefix=}")
    assert session.state.model.layer_prefix == expected_layer_prefix
