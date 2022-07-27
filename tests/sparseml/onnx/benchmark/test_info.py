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
from typing import Dict, List, Tuple

import numpy
import onnx
import pytest

from pytest_mock import MockerFixture, mocker  # noqa: F401
from sparseml.base import Framework
from sparseml.benchmark.serialization import BatchBenchmarkResult, BenchmarkResult
from sparseml.onnx.benchmark import ORTBenchmarkRunner, load_model
from sparseml.onnx.benchmark.info import load_data
from sparseml.onnx.framework import framework_info as get_framework_info
from sparsezoo import Model


TEST_STUB = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
MOCK_BENCHMARK_RETURN_VALUE = 0.5


@pytest.fixture(scope="module")
def mobilenet_fixture() -> Model:
    with tempfile.TemporaryDirectory() as onnx_dir:
        model = Model(TEST_STUB)
        model.onnx_model.download(onnx_dir)
        yield model


@pytest.fixture()
def cpu_runner_fixture(
    mocker: MockerFixture, mobilenet_fixture: Model  # noqa: F811
) -> ORTBenchmarkRunner:
    ort_model_runner = mocker.MagicMock()
    ort_model_runner.batch_forward.return_value = (
        None,
        MOCK_BENCHMARK_RETURN_VALUE,
    )
    mocker.patch(
        "sparseml.onnx.benchmark.info.ORTModelRunner", return_value=ort_model_runner
    )
    return ORTBenchmarkRunner(
        mobilenet_fixture,
        batch_size=32,
        iterations=10,
        warmup_iterations=5,
        framework_args={},
    )


class TestLoadModel:
    def test_load_model_from_sparsezoo_model(self, mobilenet_fixture: Model):
        onnx_model = onnx.load(mobilenet_fixture.onnx_model.get_path())
        assert load_model(mobilenet_fixture) == onnx_model

    def test_load_model_from_sparsezoo_file(self, mobilenet_fixture: Model):
        onnx_model = onnx.load(mobilenet_fixture.onnx_model.get_path())
        assert load_model(mobilenet_fixture.onnx_model) == onnx_model

    def test_load_model_from_stub(self, mobilenet_fixture: Model):
        onnx_model = onnx.load(mobilenet_fixture.onnx_model.get_path())
        assert (load_model(TEST_STUB, download_directory = mobilenet_fixture.get_path()) == onnx_model
        )

    def test_load_model_from_path(self, mobilenet_fixture: Model):
        onnx_model = onnx.load(mobilenet_fixture.onnx_model.get_path())
        assert load_model(mobilenet_fixture.onnx_model.get_path()) == onnx_model

    def test_load_model_from_onnx(self, mobilenet_fixture: Model):
        onnx_model = onnx.load(mobilenet_fixture.onnx_model.get_path())

        assert load_model(onnx_model) == onnx_model


class TestOrtBenchmarkRunner:
    def test_cpu_init(
        self, mocker: MockerFixture, mobilenet_fixture: Model  # noqa: F811
    ):
        model = load_model(mobilenet_fixture)
        framework_args = {}
        batch_size = 32
        iterations = 10
        warmup_iterations = 5
        ort_model_runner = mocker.MagicMock()
        ort_model_runner.batch_forward.return_value = (
            None,
            MOCK_BENCHMARK_RETURN_VALUE,
        )
        mocker.patch(
            "sparseml.onnx.benchmark.info.ORTModelRunner", return_value=ort_model_runner
        )

        runner = ORTBenchmarkRunner(
            mobilenet_fixture,
            batch_size=batch_size,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            framework_args=framework_args,
        )

        framework_info = get_framework_info()

        assert runner.framework == Framework.onnx
        assert runner.framework_info == framework_info
        assert runner.batch_size == batch_size
        assert runner.iterations == iterations
        assert runner.warmup_iterations == warmup_iterations
        assert "onnx" in runner.package_versions
        assert "onnxruntime" in runner.package_versions
        assert runner.framework_args == framework_args
        assert runner.model == model

    def test_run_batch(self, cpu_runner_fixture: ORTBenchmarkRunner):
        # Check if called with an ordered dict
        mock_data = OrderedDict([("arr00", numpy.random.randn(1, 3, 224, 224))])
        benchmark_result = cpu_runner_fixture.run_batch(mock_data)
        cpu_runner_fixture._model_runner.batch_forward.assert_called_with(mock_data)
        assert benchmark_result.batch_time == MOCK_BENCHMARK_RETURN_VALUE

        # Check if called with tuple pair of input/label
        mock_data = (mock_data, None)
        benchmark_result = cpu_runner_fixture.run_batch(mock_data)
        cpu_runner_fixture._model_runner.batch_forward.assert_called_with(mock_data[0])
        assert isinstance(benchmark_result, BatchBenchmarkResult)
        assert benchmark_result.batch_time == MOCK_BENCHMARK_RETURN_VALUE

    def test_run(
        self,
        mocker: MockerFixture,  # noqa: F811
        cpu_runner_fixture: ORTBenchmarkRunner,
    ):
        mock_data = [OrderedDict([("arr00", numpy.random.randn(3, 224, 224))])] * 5

        data_loader = load_data(
            mock_data,
            cpu_runner_fixture.model,
            batch_size=cpu_runner_fixture.batch_size,
            total_iterations=cpu_runner_fixture.iterations
            + cpu_runner_fixture.warmup_iterations,
        )
        benchmark_results = cpu_runner_fixture.run(mock_data)

        mock_calls = cpu_runner_fixture._model_runner.batch_forward.call_args_list
        for mock_call, (data, _) in zip(mock_calls, data_loader):
            mock_args, _ = mock_call
            batch_arg = mock_args[0]
            for key in data:
                assert batch_arg[key].shape == data[key].shape
                assert (batch_arg[key] == data[key]).all()

        assert (
            cpu_runner_fixture._model_runner.batch_forward.call_count
            == cpu_runner_fixture.iterations + cpu_runner_fixture.warmup_iterations
        )
        assert isinstance(benchmark_results, BenchmarkResult)
        assert len(benchmark_results.results) == cpu_runner_fixture.iterations

    def test_run_iter(
        self,
        mocker: MockerFixture,  # noqa: F811
        cpu_runner_fixture: ORTBenchmarkRunner,
    ):
        mock_data = [OrderedDict([("arr00", numpy.random.randn(3, 224, 224))])] * 5

        total_iterations = (
            cpu_runner_fixture.iterations + cpu_runner_fixture.warmup_iterations
        )
        data_loader = load_data(
            mock_data,
            cpu_runner_fixture.model,
            batch_size=cpu_runner_fixture.batch_size,
            total_iterations=total_iterations,
        )

        data_list = []
        for index, (data, _) in enumerate(data_loader):
            if index > total_iterations:
                break
            data_list.append(data)

        for index, benchmark_result in enumerate(
            cpu_runner_fixture.run_iter(mock_data)
        ):
            assert isinstance(benchmark_result, BatchBenchmarkResult)
            assert (
                cpu_runner_fixture._model_runner.batch_forward.call_count
                == index + 1 + cpu_runner_fixture.warmup_iterations
            )
            mock_args, _ = cpu_runner_fixture._model_runner.batch_forward.call_args
            batch_args = mock_args[0]
            for key in batch_args:
                assert (
                    batch_args[key]
                    == data_list[index + cpu_runner_fixture.warmup_iterations][key]
                ).all()

    def test_benchmark_config(self, cpu_runner_fixture: ORTBenchmarkRunner):
        config = cpu_runner_fixture.benchmark_config
        assert config.batch_size == cpu_runner_fixture.batch_size
        assert config.iterations == cpu_runner_fixture.iterations
        assert config.warmup_iterations == cpu_runner_fixture.warmup_iterations
        assert config.framework_args == cpu_runner_fixture.framework_args
        assert config.device == cpu_runner_fixture.device
        assert config.inference_provider == cpu_runner_fixture.inference_provider
