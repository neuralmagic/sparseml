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
from typing import NamedTuple

import pytest

from sparseml.onnx.sparsification import PruningPerformanceSensitivityAnalyzer
from sparseml.sparsification import create_pruning_recipe
from sparsezoo import Model


try:
    from sparseml.pytorch.models import mobilenet, resnet18
    from sparseml.pytorch.optim import ScheduledModifierManager
except Exception:
    mobilenet = None
    resnet18 = None
    ScheduledModifierManager = None


RECIPES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "recipes")
GENERATE_TEST_FILES = os.getenv("NM_ML_GENERATE_ORACLE_TEST_DATA", False)
GENERATE_TEST_FILES = False if GENERATE_TEST_FILES == "0" else GENERATE_TEST_FILES

OracleTestFixture = NamedTuple(
    "OracleTestFixture",
    [("generated_recipe", str), ("expected_recipe", str), ("model_lambda", str)],
)


@pytest.fixture(
    scope="session",
    params=[
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",  # noqa 501
            "mobilenet.yaml",
            mobilenet,
        ),
        (
            "zoo:cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/base-none",
            "resnet18.yaml",
            resnet18,
        ),
    ],
)
def oracle_test_params(request) -> OracleTestFixture:
    zoo_stub, expected_recipe_name, model_lambda = request.param
    zoo_model = Model(zoo_stub)
    onnx_path = zoo_model.onnx_model.path
    generated_recipe = create_pruning_recipe(
        onnx_path,
        skip_analyzer_types=[PruningPerformanceSensitivityAnalyzer],
    ).strip()

    if GENERATE_TEST_FILES:
        with open(os.path.join(RECIPES_PATH, expected_recipe_name), "w") as new_file:
            import pdb

            pdb.set_trace()
            new_file.write(generated_recipe)

    with open(os.path.join(RECIPES_PATH, expected_recipe_name)) as recipe_file:
        expected_recipe = recipe_file.read().strip()

    return OracleTestFixture(generated_recipe, expected_recipe, model_lambda)


def test_oracle_recipe_correctness(oracle_test_params):
    assert oracle_test_params.generated_recipe == oracle_test_params.expected_recipe


@pytest.mark.skipif(
    mobilenet is None or resnet18 is None,
    reason="unable to import modules from sparseml.pytorch",
)
def test_oracle_recipe_application(oracle_test_params):
    manager = ScheduledModifierManager.from_yaml(oracle_test_params.generated_recipe)
    model = oracle_test_params.model_lambda()
    manager.apply(model)
    assert model is not None
