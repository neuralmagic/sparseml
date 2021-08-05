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
from collections import OrderedDict
from typing import Any, Dict

import numpy
import onnx
import pytest

from pytest_mock import MockerFixture, mocker  # noqa: F401
from sparseml.base import Framework
from sparseml.onnx.benchmark import (
    ORTBenchmarkRunner,
    ORTCpuBenchmarkRunner,
    ORTCudaBenchmarkRunner,
    create_benchmark_runner,
    detect_benchmark_runner,
    load_model,
)
from sparsezoo.models import Zoo
from sparsezoo.objects import Model


TEST_STUB = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
MOCK_BENCHMARK_RETURN_VALUE = 0.5


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


@pytest.mark.parametrize(
    (
        "batch_size,iterations,warmup_iterations,provider,device,framework_args,"
        "ort_execution_provider,ort_benchmark_class"
    ),
    [
        (32, 100, 10, "cpu", "cpu", {}, "CPUExecutionProvider", ORTCpuBenchmarkRunner),
    ],
)
def test_create_benchmark_runner(
    mocker: MockerFixture,  # noqa: F811
    mobilenet_fixture: Model,
    batch_size: int,
    iterations: int,
    warmup_iterations: int,
    provider: str,
    device: str,
    framework_args: Dict[str, Any],
    ort_execution_provider: str,
    ort_benchmark_class: type,
):
    ort_model_runner = mocker.MagicMock()
    mocker.patch("sparseml.onnx.benchmark.info.ORTModelRunner", ort_model_runner)
    runner = create_benchmark_runner(
        mobilenet_fixture,
        batch_size=batch_size,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        provider=provider,
        device=device,
        framework_args=framework_args,
    )

    assert isinstance(runner, ort_benchmark_class)
    ort_model_runner.assert_called_with(
        runner.model,
        batch_size=batch_size,
        providers=[ort_execution_provider],
        **framework_args,
    )
    assert runner.framework == Framework.onnx
    assert runner.batch_size == batch_size
    assert runner.iterations == iterations
    assert runner.warmup_iterations == warmup_iterations
    assert runner.device == device
    assert runner.inference_provider.name == provider
    assert runner.framework_args == framework_args
    assert runner.inference_provider.name == provider
    assert runner.inference_provider.device == device


class TestOrtBenchmarkRunner:
    @pytest.fixture()
    def cpu_benchmark_runner(
        self, mocker: MockerFixture, mobilenet_fixture: Model  # noqa: F811
    ) -> ORTBenchmarkRunner:
        ort_model_runner = mocker.MagicMock()
        ort_model_runner.batch_forward.return_value = (
            None,
            MOCK_BENCHMARK_RETURN_VALUE,
        )
        mocker.patch(
            "sparseml.onnx.benchmark.info.ORTModelRunner", return_value=ort_model_runner
        )
        return ORTCpuBenchmarkRunner(
            mobilenet_fixture,
            batch_size=32,
            iterations=10,
            warmup_iterations=10,
            framework_args={},
        )

    def test_run_batch(self, cpu_benchmark_runner: ORTBenchmarkRunner):
        # Check if called with an ordered dict
        mock_data = OrderedDict([("arr00", numpy.random.randn(1, 3, 224, 224))])
        benchmark_result = cpu_benchmark_runner.run_batch(mock_data)
        cpu_benchmark_runner._model_runner.batch_forward.assert_called_with(mock_data)
        assert benchmark_result.batch_time == MOCK_BENCHMARK_RETURN_VALUE

        # Check if called with tuple pair of input/label
        mock_data = (mock_data, None)
        benchmark_result = cpu_benchmark_runner.run_batch(mock_data)
        cpu_benchmark_runner._model_runner.batch_forward.assert_called_with(
            mock_data[0]
        )
        assert benchmark_result.batch_time == MOCK_BENCHMARK_RETURN_VALUE

    def test_run_iterable(self, cpu_benchmark_runner: ORTBenchmarkRunner):
        mock_data_single = [
            OrderedDict([("arr00", numpy.random.randn(1, 3, 224, 224))])
        ]
        mock_data = mock_data_single * cpu_benchmark_runner.batch_size
        benchmark_results = cpu_benchmark_runner.run(mock_data)

        mock_calls = cpu_benchmark_runner._model_runner.batch_forward.call_args_list
        for mock_call in mock_calls:
            mock_args, _ = mock_call
            for arg, data in zip(mock_args, mock_data_single):
                for key in data:
                    assert numpy.all(arg[key] == data[key])

        assert (
            cpu_benchmark_runner._model_runner.batch_forward.call_count
            == cpu_benchmark_runner.iterations + cpu_benchmark_runner.warmup_iterations
        )
        assert len(benchmark_results.results) == cpu_benchmark_runner.iterations
