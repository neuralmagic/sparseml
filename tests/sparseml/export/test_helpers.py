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
from collections import OrderedDict

import onnx
import pytest

from src.sparseml.export.helpers import apply_optimizations
from tests.sparseml.exporters.transforms.test_onnx_transform import (
    _create_model as create_dummy_onnx_file,
)


def foo(onnx_model):
    logging.debug("foo")
    return onnx_model


def bar(onnx_model):
    logging.debug("bar")
    return onnx_model


@pytest.fixture()
def available_optimizations():
    return OrderedDict(zip(["bar", "foo"], [bar, foo]))


@pytest.fixture()
def available_optimizations_empty():
    return OrderedDict()


@pytest.mark.parametrize(
    "target_optimizations, should_raise_error",
    [("none", False), ("all", False), ("error_name", True), (["error_name"], True)],
)
def test_apply_optimizations_empty(
    tmp_path, available_optimizations_empty, target_optimizations, should_raise_error
):
    onnx_model = create_dummy_onnx_file()
    onnx_file_path = tmp_path / "test.onnx"
    onnx.save(onnx_model, onnx_file_path)

    if not should_raise_error:
        apply_optimizations(
            onnx_file_path=onnx_file_path,
            target_optimizations=target_optimizations,
            available_optimizations=available_optimizations_empty,
        )
    else:
        with pytest.raises(KeyError):
            apply_optimizations(
                onnx_file_path=onnx_file_path,
                target_optimizations=target_optimizations,
                available_optimizations=available_optimizations_empty,
            )


@pytest.mark.parametrize(
    "target_optimizations, expected_logs, should_raise_error",
    [
        ("none", [], False),
        ("all", ["bar", "foo"], False),
        (["foo"], ["foo"], False),
        ("error_name", [], True),
        (["error_name"], [], True),
    ],
)
def test_apply_optimizations(
    caplog,
    tmp_path,
    available_optimizations,
    target_optimizations,
    expected_logs,
    should_raise_error,
):
    onnx_model = create_dummy_onnx_file()
    onnx_file_path = tmp_path / "test.onnx"
    onnx.save(onnx_model, onnx_file_path)

    if should_raise_error:
        with pytest.raises(KeyError):
            apply_optimizations(
                onnx_file_path=onnx_file_path,
                target_optimizations=target_optimizations,
                available_optimizations=available_optimizations,
            )
        return

    with caplog.at_level(logging.DEBUG):
        apply_optimizations(
            onnx_file_path=onnx_file_path,
            target_optimizations=target_optimizations,
            available_optimizations=available_optimizations,
        )

        assert caplog.messages == expected_logs
