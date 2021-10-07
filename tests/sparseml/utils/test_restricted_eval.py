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

from sparseml.utils import UnknownVariableException, restricted_eval


@pytest.mark.parametrize(
    "expression,variables,expected_result",
    [
        ("1", {}, 1),
        ("2 *3 + 1", {}, 7),
        ("2 * (3 + 1)", {}, 8),
        ("x", {"w": 1, "x": 5.2, "y": 3.1}, 5.2),
        ("0.2 * num_epochs", {"num_epochs": 100}, 20),
        ("num_epochs * 0.2", {"num_epochs": 100}, 20),
        ("num_epochs + offset", {"num_epochs": 100, "offset": 5}, 105),
        (
            "min(0.2 * num_epochs, pruning_start) - 5",
            {"num_epochs": 100, "pruning_start": 19},
            14,
        ),
        (
            "min(0.2 * num_epochs, pruning_start) - 5",
            {"num_epochs": 100, "pruning_start": 21},
            15,
        ),
    ],
)
def test_restricted_eval(expression, variables, expected_result):
    assert restricted_eval(expression, variables) == expected_result


@pytest.mark.parametrize(
    "expression,variables,expected_exception",
    [
        ("5 + num_epochs", {}, UnknownVariableException),
        ("5 + num_epochs", {"x": 5}, UnknownVariableException),
        ("num_epochs + offset", {"num_epochs": 100}, UnknownVariableException),
        ("[1,2]", {}, RuntimeError),
        ("tuple(5)", {}, RuntimeError),
    ],
)
def test_restricted_eval_exceptions(expression, variables, expected_exception):
    with pytest.raises(expected_exception):
        restricted_eval(expression, variables)
