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

from unittest.mock import MagicMock

import pytest

from sparseml.evaluation.evaluator import evaluate
from sparsezoo.evaluation.results import Result


def test_evaluate_calls_integration_with_correct_parameters(mocker):
    mock_integration = MagicMock(return_value=Result(formatted=[], raw=None))
    mocker.patch(
        "sparseml.evaluation.evaluator.SparseMLEvaluationRegistry.resolve",
        return_value=mock_integration,
    )
    model_path = "model_path"
    datasets = "dataset_name"
    integration = "integration_name"
    batch_size = 10

    result = evaluate(
        model_path=model_path,
        datasets=datasets,
        integration=integration,
        batch_size=batch_size,
    )

    assert isinstance(result, Result)


def test_evaluate_raises_value_error_for_unregistered_integration():
    with pytest.raises(ValueError, match="No registered integrations"):
        evaluate(
            model_path="model_path",
            integration="foo",
        )
