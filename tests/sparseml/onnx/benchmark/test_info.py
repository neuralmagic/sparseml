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

import tempfile

import onnx
import pytest

from sparseml.onnx.benchmark import (
    ORTBenchmarkRunner,
    ORTCpuBenchmarkRunner,
    ORTCudaBenchmarkRunner,
    detect_benchmark_runner,
    load_model,
)
from sparsezoo.models import Zoo
from sparsezoo.objects import Model


TEST_STUB = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"


@pytest.fixture(scope="module")
def mobilenet_fixture() -> Model:
    with tempfile.TemporaryDirectory() as onnx_dir:
        yield Zoo.download_model_from_stub(TEST_STUB, override_parent_path=onnx_dir)


def test_load_model_from_sparsezoo_model(mobilenet_fixture: Model):
    onnx_model = onnx.load(mobilenet_fixture.onnx_file.path)
    assert load_model(mobilenet_fixture) == onnx_model


def test_load_model_from_sparsezoo_file(mobilenet_fixture):
    onnx_model = onnx.load(mobilenet_fixture.onnx_file.path)
    assert load_model(mobilenet_fixture.onnx_file) == onnx_model


def test_load_model_from_stub(mobilenet_fixture):
    onnx_model = onnx.load(mobilenet_fixture.onnx_file.path)
    assert (
        load_model(
            TEST_STUB, override_parent_path=mobilenet_fixture.override_parent_path
        )
        == onnx_model
    )


def test_load_model_from_path(mobilenet_fixture):
    onnx_model = onnx.load(mobilenet_fixture.onnx_file.path)
    assert load_model(mobilenet_fixture.onnx_file.path) == onnx_model


def test_load_model_from_onnx(mobilenet_fixture):
    mobilenet_fixture.onnx_file.download()
    onnx_model = onnx.load(mobilenet_fixture.onnx_file.path)

    assert load_model(onnx_model) == onnx_model


@pytest.mark.parametrize(
    "provider,device,benchmark_class",
    [
        ("cpu", "cpu", ORTCpuBenchmarkRunner),
        ("cuda", "gpu", ORTCudaBenchmarkRunner),
        ("vnni", "cpu", ORTBenchmarkRunner),
    ],
)
def test_detect_benchmark_runner(provider: str, device: str, benchmark_class: type):
    assert detect_benchmark_runner(provider=provider, device=device) == benchmark_class
