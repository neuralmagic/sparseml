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
import tarfile
from collections import OrderedDict

import onnx
import pytest

from src.sparseml.export.helpers import (
    apply_optimizations,
    export_sample_inputs_outputs,
)
from tests.sparseml.exporters.transforms.test_onnx_transform import (
    _create_model as create_dummy_onnx_file,
)


@pytest.mark.parametrize(
    "as_tar",
    [True, False],
)
def test_export_sample_inputs_outputs(tmp_path, as_tar):
    pytest.importorskip("torch", reason="test requires pytorch")
    import torch

    batch_size = 3
    num_samples = 5

    input_samples = [torch.randn(batch_size, 3, 224, 224) for _ in range(num_samples)]
    output_samples = [torch.randn(batch_size, 1000) for _ in range(num_samples)]

    export_sample_inputs_outputs(
        input_samples=input_samples,
        output_samples=output_samples,
        target_path=tmp_path,
        as_tar=as_tar,
    )
    dir_names = {"sample-inputs", "sample-outputs"}
    dir_names_tar = {"sample-inputs.tar.gz", "sample-outputs.tar.gz"}

    if as_tar:
        assert set(os.listdir(tmp_path)) == dir_names_tar
        # unpack the tar files
        for dir_name in dir_names_tar:
            with tarfile.open(os.path.join(tmp_path, dir_name)) as tar:
                tar.extractall(path=tmp_path)

    assert set(os.listdir(tmp_path)) == (
        dir_names if not as_tar else dir_names_tar | dir_names
    )
    assert set(os.listdir(os.path.join(tmp_path, "sample-inputs"))) == {
        "inp-0000.npz",
        "inp-0001.npz",
        "inp-0002.npz",
        "inp-0003.npz",
        "inp-0004.npz",
    }
    assert set(os.listdir(os.path.join(tmp_path, "sample-outputs"))) == {
        "out-0000.npz",
        "out-0001.npz",
        "out-0002.npz",
        "out-0003.npz",
        "out-0004.npz",
    }


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
