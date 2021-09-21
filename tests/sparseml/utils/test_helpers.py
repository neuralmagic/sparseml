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

from sparseml.utils import (
    ALL_TOKEN,
    convert_to_bool,
    flatten_iterable,
    interpolate,
    validate_str_iterable,
)


@pytest.mark.parametrize(
    "test_list,output",
    [
        ([], []),
        ([0, 1], [0, 1]),
        ([[0, 1], [2, 3]], [0, 1, 2, 3]),
        ([[0, 1], 2, 3], [0, 1, 2, 3]),
    ],
)
def test_flatten_iterable(test_list, output):
    flattened = flatten_iterable(test_list)
    assert flattened == output


@pytest.mark.parametrize(
    "test_bool,output",
    [
        (True, True),
        ("t", True),
        ("T", True),
        ("true)", True),
        ("True", True),
        (1, True),
        ("1", True),
        (False, False),
        ("f", False),
        ("F", False),
        ("false", False),
        ("False", False),
        (0, False),
        ("0", False),
    ],
)
def test_convert_to_bool(test_bool, output):
    converted = convert_to_bool(test_bool)
    assert converted == output


@pytest.mark.parametrize(
    "test_list,output",
    [
        (ALL_TOKEN, ALL_TOKEN),
        (ALL_TOKEN.lower(), ALL_TOKEN),
        ([], []),
        ([0, 1], [0, 1]),
        ([[0], [1]], [0, 1]),
    ],
)
def test_validate_str_iterable(test_list, output):
    validated = validate_str_iterable(test_list, "")
    assert validated == output


def test_validate_str_iterable_negative():
    with pytest.raises(ValueError):
        validate_str_iterable("will fail", "")


@pytest.mark.parametrize(
    "x_cur,x0,x1,y0,y1,inter_func,out",
    [
        (0.0, 0.0, 1.0, 0.0, 5.0, "linear", 0.0),
        (0.0, 0.0, 1.0, 0.0, 5.0, "cubic", 0.0),
        (0.0, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 0.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "linear", 5.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "cubic", 5.0),
        (1.0, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 5.0),
        (0.5, 0.0, 1.0, 0.0, 5.0, "linear", 2.5),
        (0.5, 0.0, 1.0, 0.0, 5.0, "cubic", 4.375),
        (0.5, 0.0, 1.0, 0.0, 5.0, "inverse_cubic", 1.031),
    ],
)
def test_interpolate(x_cur, x0, x1, y0, y1, inter_func, out):
    interpolated = interpolate(x_cur, x0, x1, y0, y1, inter_func)
    assert abs(out - interpolated) < 0.01
