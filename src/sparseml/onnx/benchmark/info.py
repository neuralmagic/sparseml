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
from typing import Any, Callable, Dict, Iterable, Optional

import onnx
from onnx import ModelProto

from sparseml.base import Framework
from sparseml.benchmark import (
    BatchBenchmarkResult,
    BenchmarkInfo,
    BenchmarkResults,
    BenchmarkRunner,
)
from sparseml.framework import FrameworkInfo
from sparseml.framework.info import FrameworkInferenceProviderInfo
from sparseml.onnx.base import (
    require_onnx,
    require_onnxruntime,
    require_onnxruntime_gpu,
)
from sparseml.onnx.framework import framework_info as get_framework_info
from sparseml.onnx.framework import is_supported
from sparseml.onnx.utils import DataLoader, ORTModelRunner
from sparsezoo.models import Zoo
from sparsezoo.objects import File, Model
from sparsezoo.utils import DataLoader as SparseZooDataLoader
from sparsezoo.utils import Dataset as SparseZooDataset
from typing_extensions import OrderedDict


__all__ = [
    "ORTBenchmarkRunner",
    "ORTCpuBenchmarkRunner",
    "ORTCudaBenchmarkRunner",
    "load_model",
    "load_data",
    "detect_benchmark_runner",
    "create_benchmark_runner",
    "run_benchmark",
]

_LOGGER = logging.getLogger(__name__)


class ORTBenchmarkRunner(BenchmarkRunner):
    """
    Benchmark runner for ONNXruntime.

    :param model: model to benchmark
    :param batch_size: batch size to use for benchmarking
    :param iterations: number of iterations to run
    :param warmup_iterations: number of warmup iterations to run
    :param framework_args: additional arguments to pass to the framework
    :param provider: inference provider name to use from available
        FrameworkInfo
    :param device: device to use for benchmarking
    :param ort_provider: provider to use for ONNXruntime
    """

    @require_onnx()
    @require_onnxruntime()
    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        iterations: int = 0,
        warmup_iterations: int = 0,
        framework_args: Dict[str, Any] = {},
        provider: str = "cpu",
        device: str = "cpu",
        ort_provider: str = "CPUExecutionProvider",
        **kwargs,
    ):
        if iterations < 0:
            raise ValueError(
                "iterations must be non-negative, where 0 will run entire dataset."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative.")

        self._model = load_model(model)

        self._framework_info = get_framework_info()
        self._package_versions = self._framework_info.package_versions
        inference_providers = [
            inference_provider
            for inference_provider in self._framework_info.inference_providers
            if inference_provider.name == provider
            and inference_provider.device == device
        ]
        if len(inference_providers) == 0:
            raise ValueError(f"No supported inference provider found for {provider}.")

        if ort_provider not in self._framework_info.properties["available_providers"]:
            raise ValueError(f"Provider {ort_provider} not installed.")

        self._model_runner = ORTModelRunner(
            self._model,
            batch_size=batch_size,
            providers=[ort_provider],
            **framework_args,
        )

        self._inference_provider = inference_providers[0]
        self._provider = provider
        self._device = device
        self._framework_args = framework_args

        self._batch_size = batch_size
        self._iterations = iterations
        self._warmup_iterations = warmup_iterations

    def run_iter(
        self,
        data: Any,
        desc: str = "",
        show_progress: bool = False,
        *args,
        **kwargs,
    ) -> BenchmarkResults:
        """
        Runs a benchmark on the given data.

        :param data: data to use for benchmarking
        :param show_progress: whether to show progress
        :param args: additional arguments to pass to the framework
        :param kwargs: additional arguments to pass to the framework
        """
        data = load_data(
            data,
            self._model,
            self._batch_size,
            self._warmup_iterations + self._iterations,
        )
        return super().run_iter(
            data, desc=desc, show_progress=show_progress, *args, **kwargs
        )

    def run_batch(self, batch: Any, *args, **kwargs) -> BatchBenchmarkResult:
        if isinstance(batch, tuple):
            batch = batch[0]
        outputs, batch_time = self._model_runner.batch_forward(batch, *args, **kwargs)
        return BatchBenchmarkResult(
            batch_time, self._batch_size, inputs=batch, outputs=outputs
        )

    @property
    def framework(self) -> Framework:
        """
        :return: the framework
        """
        return Framework.onnx

    @property
    def framework_info(self) -> FrameworkInfo:
        """
        :return: the framework info
        """
        return self._framework_info

    @property
    def batch_size(self) -> int:
        """
        :return: the batch size
        """
        return self._batch_size

    @property
    def warmup_iterations(self) -> int:
        """
        :return: the warmup iterations
        """
        return self._warmup_iterations

    @property
    def iterations(self) -> int:
        """
        :return: the number of iterations
        """
        return self._iterations

    @property
    def inference_provider(self) -> FrameworkInferenceProviderInfo:
        """
        :return: the inference provider
        """
        return self._inference_provider

    @property
    def package_versions(self) -> Dict[str, str]:
        """
        :return: the package versions
        """
        return self._package_versions

    @property
    def framework_args(self) -> Dict[str, Any]:
        """
        :return: the framework args
        """
        return self._framework_args

    @property
    def device(self) -> str:
        """
        :return: the device
        """
        return self._device


class ORTCudaBenchmarkRunner(ORTBenchmarkRunner):
    """
    Benchmark runner for ONNXruntime on CUDA. Will create an
    ORTBenchmarkRunner using the CUDAExecutionProvider and with
    provider and device set to cuda and gpu respectively.

    :param model: model to benchmark
    :param batch_size: batch size
    :param iterations: number of iterations to run
    :param warmup_iterations: number of warmup iterations
    :param framework_args: additional arguments for the framework
    """

    PROVIDER: str = "cuda"
    DEVICE: str = "gpu"

    @require_onnxruntime_gpu()
    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        iterations: int = 0,
        warmup_iterations: int = 0,
        framework_args: Dict[str, Any] = {},
        **kwargs,
    ):
        del kwargs["provider"]
        del kwargs["device"]
        super().__init__(
            model=model,
            batch_size=batch_size,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            framework_args=framework_args,
            ort_provider="CUDAExecutionProvider",
            provider=ORTCudaBenchmarkRunner.PROVIDER,
            device=ORTCudaBenchmarkRunner.DEVICE,
            **kwargs,
        )


class ORTCpuBenchmarkRunner(ORTBenchmarkRunner):
    """
    Benchmark runner for ONNXruntime on CPU. Will create an
    ORTBenchmarkRunner using the CPUExecutionProvider and with
    provider and device set to cpu and cpu respectively.

    :param model: model to benchmark
    :param batch_size: batch size
    :param iterations: number of iterations to run
    :param warmup_iterations: number of warmup iterations
    :param framework_args: additional arguments for the framework
    """

    PROVIDER: str = "cpu"
    DEVICE: str = "cpu"

    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        iterations: int = 0,
        warmup_iterations: int = 0,
        framework_args: Dict[str, Any] = {},
        **kwargs,
    ):
        del kwargs["provider"]
        del kwargs["device"]
        super().__init__(
            model=model,
            batch_size=batch_size,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            framework_args=framework_args,
            ort_provider="CPUExecutionProvider",
            provider=ORTCpuBenchmarkRunner.PROVIDER,
            device=ORTCpuBenchmarkRunner.DEVICE,
            **kwargs,
        )


def load_model(model: Any, **kwargs) -> ModelProto:
    """
    Loads the model and saves it to a temporary file if necessary

    :param model: the model
    :param kwargs: additional arguments to pass if loading from a stub
    :return: the model loaded as a ModelProto
    """
    if not model:
        raise ValueError("Model must not be None type")

    if isinstance(model, str) and model.startswith("zoo:"):
        model = Zoo.load_model_from_stub(model, **kwargs)

    if isinstance(model, Model):
        # default to the main onnx file for the model
        model = model.onnx_file.downloaded_path()
    elif isinstance(model, File):
        # get the downloaded_path -- will auto download if not on local system
        model = model.downloaded_path()
    elif isinstance(model, ModelProto):
        return model

    if not isinstance(model, str):
        raise ValueError("unsupported type for model: {}".format(type(model)))

    if not os.path.exists(model):
        raise ValueError("model path must exist: given {}".format(model))

    return onnx.load(model)


def load_data(
    data: Any, model: Any, batch_size: int, total_iterations: int
) -> Iterable[OrderedDict[str, Any]]:
    """
    Creates a DataLoader for the given data. If data is None value,
    then random data will be generated using the given model.

    :param data: data to use for benchmarking
    :param model: model to use for generating data
    :param batch_size: batch size
    :param total_iterations: total number of iterations
    :return: DataLoader
    """
    if not data:
        model = load_model(model)
        return DataLoader.from_model_random(
            model, batch_size, iter_steps=total_iterations
        )

    if isinstance(data, str) and data.startswith("zoo:"):
        model_from_zoo = Zoo.load_model_from_stub(data)
        return model_from_zoo.data_inputs.loader(
            batch_size, total_iterations, batch_as_list=False
        )

    if isinstance(data, DataLoader):
        return data

    if isinstance(data, SparseZooDataLoader):
        datasets = [
            SparseZooDataset(name, dataset) for name, dataset in data.datasets.items()
        ]
        data = SparseZooDataLoader(*datasets, batch_size=1, batch_as_list=False)
        data = [
            OrderedDict(
                [
                    (element, value.reshape(value.shape[1:]))
                    for element, value in entry.items()
                ]
            )
            for entry in data
        ]
        return DataLoader(
            data, None, batch_size=batch_size, iter_steps=total_iterations
        )

    if isinstance(data, str) or isinstance(data, Iterable):
        return DataLoader(
            data, None, batch_size=batch_size, iter_steps=total_iterations
        )

    return data


def detect_benchmark_runner(
    provider: Optional[str] = None,
    device: Optional[str] = None,
) -> Callable[
    [Any, int, int, int, Dict[str, Any], Optional[str], Optional[str]],
    ORTBenchmarkRunner,
]:
    """
    Detects the benchmark runner based on the provider and device.

    :param provider: inference provider name to use from available
        FrameworkInfo
    :param device: the device to use for benchmarking
    :return: callable for contsructing the benchmark runner
    """
    # Obtains the provider/device name if any of them are not provided
    if provider is None or device is None:
        framework_info = get_framework_info()
        if len(framework_info.inference_providers) < 1:
            raise RuntimeError(
                "No inference providers available. Please install "
                "onnxruntime or onnx pip."
            )

        if provider is None and device is None:
            provider = framework_info.inference_providers[0].name
            device = framework_info.inference_providers[0].device
        elif provider is None:
            matching_provider = [
                inference_provider
                for inference_provider in framework_info.inference_providers
                if inference_provider.device == device
            ]
            if len(matching_provider) == 0:
                raise ValueError(
                    f"No inference providers available for device {device}."
                )
            provider = matching_provider[0].name
        elif device is None:
            matching_provider = [
                inference_provider
                for inference_provider in framework_info.inference_providers
                if inference_provider.name == provider
            ]
            if len(matching_provider) == 0:
                raise ValueError(
                    f"No inference providers available for provider {provider}."
                )
            device = matching_provider[0].device

    if (
        provider == ORTCpuBenchmarkRunner.PROVIDER
        and device == ORTCpuBenchmarkRunner.DEVICE
    ):
        return ORTCpuBenchmarkRunner
    elif (
        provider == ORTCudaBenchmarkRunner.PROVIDER
        and device == ORTCudaBenchmarkRunner.DEVICE
    ):
        return ORTCudaBenchmarkRunner
    else:
        return ORTBenchmarkRunner


def create_benchmark_runner(
    model: Any,
    batch_size: int = 1,
    iterations: int = 0,
    warmup_iterations: int = 0,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
    framework_args: Dict[str, Any] = {},
    **kwargs,
) -> BenchmarkRunner:
    """
    Create a benchmark runner for the given model.

    :param model: Model to benchmark.
    :param batch_size: Batch size to use for the benchmark.
    :param iterations: number of iterations to run the model.
    :param warmup_iterations: number of warmup iterations to run the model.
    ::param provider: inference provider name to use from available
        FrameworkInfo
    :param device: Device to use for inference.
    :param show_progress: Show progress bar.
    :param framework_args: Additional arguments to pass to the framework.
    :return: Benchmark runner for the given model for the given provider and device.
    """
    runner_constructor = detect_benchmark_runner(provider, device)
    return runner_constructor(
        model,
        batch_size=batch_size,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        framework_args=framework_args,
        show_progress=show_progress,
        provider=provider,
        device=device,
        **kwargs,
    )


@require_onnx()
@require_onnxruntime()
def run_benchmark(
    model: Any,
    data: Any = None,
    batch_size: int = 1,
    iterations: int = 0,
    warmup_iterations: int = 0,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    framework_args: Dict[str, Any] = {},
    show_progress: bool = True,
    **kwargs,
) -> BenchmarkInfo:
    """
    Run a benchmark for the given model.

    :param model: model to benchmark
    :param data: data to benchmark
    :param batch_size: batch size
    :param iterations: number of iterations
    :param warmup_iterations: number of warmup iterations
    :param framework: the specific framework run the benchmark in
    :param provider: the specific inference provider to use
    :param device: the specific device to use
    :param save_path: path to save the benchmark results
    :param framework_args: additional framework specific arguments to
        pass to the runner
    :param show_progress: True to show a tqdm bar when running, False otherwise
    :param kwargs: Additional arguments to pass to the framework.
    :return: BenchmarkInfo
    """
    if data is None and isinstance(model, str) and model.startswith("zoo:"):
        data = model

    model = load_model(model)

    if is_supported(model):
        benchmark_runner = create_benchmark_runner(
            model,
            batch_size=batch_size,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            provider=provider,
            device=device,
            framework_args=framework_args,
            show_progress=show_progress,
            **kwargs,
        )
        results = benchmark_runner.run(data, show_progress=show_progress)

        return BenchmarkInfo(
            framework=benchmark_runner.framework,
            package_versions=benchmark_runner.package_versions,
            benchmark=results.dict(),
            config=benchmark_runner.benchmark_config,
        )
    else:
        raise ValueError(
            "Model is not supported by the onnxruntime backend. "
            "Please check the model for support."
        )
