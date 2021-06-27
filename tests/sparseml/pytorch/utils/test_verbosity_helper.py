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

from sparseml.pytorch.utils.verbosity_helper import Verbosity


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (1, Verbosity.DEFAULT),
        (2, Verbosity.ON_LR_CHANGE),
        (3, Verbosity.ON_EPOCH_CHANGE),
        (4, Verbosity.ON_LR_OR_EPOCH_CHANGE),
        (True, Verbosity.DEFAULT),
        (0, Verbosity.OFF),
        (False, Verbosity.OFF),
    ],
)
def test_convert_int_to_verbosity(test_input, expected):
    assert Verbosity.convert_int_to_verbosity(test_input) == expected


@pytest.mark.parametrize(
    "test_input",
    [
        -1,
        float("inf"),
        "invalid_inp",
    ],
)
def test_exception(test_input):
    with pytest.raises(ValueError):
        assert Verbosity.convert_int_to_verbosity(test_input)
