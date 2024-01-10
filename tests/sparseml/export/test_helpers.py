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
import os
from collections import OrderedDict

import onnx
import pytest

from sparseml.export.helpers import apply_optimizations, create_deployment_folder
from tests.sparseml.exporters.transforms.test_onnx_transform import (
    _create_model as create_dummy_onnx_file,
)


def create_files(source_path, target_path):
    # create model.onnx
    model_onnx_path = target_path / "model.onnx"
    model_onnx_path.touch()

    # create model.data
    model_data_path = target_path / "model.data"
    model_data_path.touch()

    # create dummy_file
    dummy_file_path = source_path / "dummy_file"
    dummy_file_path.touch()

    # create dummy_directory
    dummy_directory_path = source_path / "dummy_directory"
    dummy_directory_path.mkdir()


@pytest.fixture()
def create_files_func():
    return create_files


@pytest.mark.parametrize(
    "deployment_directory_list, expected_deployment_file_names",
    [
        (
            ["model.onnx", "dummy_file", "dummy_directory"],
            {"model.onnx", "model.data", "dummy_file", "dummy_directory"},
        ),
    ],
)
def test_create_deployment_folder(
    tmp_path,
    deployment_directory_list,
    expected_deployment_file_names,
    create_files_func,
):
    source_path = tmp_path / "source"
    source_path.mkdir()

    target_path = tmp_path / "target"
    target_path.mkdir()

    create_files_func(source_path, target_path)

    create_deployment_folder(
        source_path=source_path,
        target_path=target_path,
        deployment_directory_files_mandatory=deployment_directory_list,
    )

    assert (
        set(os.listdir(os.path.join(target_path, "deployment")))
        == expected_deployment_file_names
    )


def foo(*args, **kwargs):
    logging.debug("foo applied")
    return True


def bar(*args, **kwargs):
    logging.debug("bar applied")
    return True


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
        ("all", ["bar applied", "foo applied"], False),
        (["foo"], ["foo applied"], False),
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

        assert set(expected_logs).issubset(set(caplog.messages))
