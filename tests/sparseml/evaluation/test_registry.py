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

from pathlib import Path

import pytest

from sparseml.evaluation.registry import (
    SparseMLEvaluationRegistry,
    collect_integrations,
)


@pytest.fixture()
def empty_eval_registry():
    from sparsezoo.utils.registry import _ALIAS_REGISTRY, _REGISTRY

    _REGISTRY[SparseMLEvaluationRegistry].clear()
    _ALIAS_REGISTRY[SparseMLEvaluationRegistry].clear()
    yield


def test_resolve_returns_value_from_registry(mocker, empty_eval_registry):
    mock_collect_integrations = mocker.patch(
        "sparseml.evaluation.registry.collect_integrations"
    )
    mock_get_value_from_registry = mocker.patch.object(
        SparseMLEvaluationRegistry, "get_value_from_registry", return_value="value"
    )
    name = "integration_name"

    result = SparseMLEvaluationRegistry.resolve(name)

    mock_collect_integrations.assert_called_once()
    mock_get_value_from_registry.assert_called_once_with(name=name)
    assert result == "value"


def _collect_dummy_integration(integration_name="dummy"):
    dummy_config = Path(__file__).parent / "dummy_config.yaml"
    collect_integrations(integration_name, dummy_config)


def test_collect(empty_eval_registry):
    integration_name = "dummy"
    _collect_dummy_integration(integration_name=integration_name)
    assert integration_name in SparseMLEvaluationRegistry.registered_names()


def test_resolve(empty_eval_registry):
    integration_name = "dummy"
    from .dummy_integration import dummy_integration

    actual_callable = SparseMLEvaluationRegistry.resolve(
        integration_name,
        integration_config_path=Path(__file__).parent / "dummy_config.yaml",
    )
    assert actual_callable == dummy_integration
