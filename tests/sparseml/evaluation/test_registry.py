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

from sparseml.evaluation.registry import (
    SparseMLEvaluationRegistry,
    collect_integrations,
)


def test_collect_integrations_logs_message_for_invalid_name(caplog):
    name = "invalid_integration_name"
    with caplog.at_level(logging.DEBUG, logger="sparseml.evaluation.registry"):
        collect_integrations(name)
    assert f"Auto collection of {name} integration for eval failed." in caplog.text


def test_resolve_returns_value_from_registry(mocker):
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
