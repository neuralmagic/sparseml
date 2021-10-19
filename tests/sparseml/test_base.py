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

from typing import Any

import pytest

from sparseml import __version__
from sparseml.base import (
    Framework,
    check_version,
    detect_framework,
    execute_in_sparseml_framework,
    get_version,
)


def test_framework():
    assert len(Framework) == 6
    assert Framework.unknown
    assert Framework.deepsparse
    assert Framework.onnx
    assert Framework.keras
    assert Framework.pytorch
    assert Framework.tensorflow_v1


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("unknown", Framework.unknown),
        ("deepsparse", Framework.deepsparse),
        ("onnx", Framework.onnx),
        ("keras", Framework.keras),
        ("pytorch", Framework.pytorch),
        ("tensorflow_v1", Framework.tensorflow_v1),
        (Framework.unknown, Framework.unknown),
        (Framework.deepsparse, Framework.deepsparse),
        (Framework.onnx, Framework.onnx),
        (Framework.keras, Framework.keras),
        (Framework.pytorch, Framework.pytorch),
        (Framework.tensorflow_v1, Framework.tensorflow_v1),
    ],
)
def test_detect_framework(inp: Any, expected: Framework):
    detected = detect_framework(inp)
    assert detected == expected


def test_execute_in_sparseml_framework():
    with pytest.raises(ValueError):
        execute_in_sparseml_framework(Framework.unknown, "unknown")

    with pytest.raises(Exception):
        execute_in_sparseml_framework(Framework.onnx, "unknown")

    # TODO: fill in with sample functions to execute in frameworks once available


def test_get_version():
    version = get_version(
        "sparseml", raise_on_error=True, alternate_package_names=["sparseml-nightly"]
    )
    assert version == __version__

    with pytest.raises(ImportError):
        get_version("unknown", raise_on_error=True)

    assert not get_version("unknown", raise_on_error=False)


def test_check_version():
    assert check_version("sparseml", alternate_package_names=["sparseml-nightly"])

    assert not check_version(
        "sparseml",
        min_version="10.0.0",
        raise_on_error=False,
        alternate_package_names=["sparseml-nightly"],
    )
    with pytest.raises(ImportError):
        check_version(
            "sparseml",
            min_version="10.0.0",
            alternate_package_names=["sparseml-nightly"],
        )

    assert not check_version(
        "sparseml",
        max_version="0.0.1",
        raise_on_error=False,
        alternate_package_names=["sparseml-nightly"],
    )
    with pytest.raises(ImportError):
        check_version(
            "sparseml",
            max_version="0.0.1",
            alternate_package_names=["sparseml-nightly"],
        )

    assert not check_version("unknown", raise_on_error=False)
    with pytest.raises(ImportError):
        check_version("unknown")
