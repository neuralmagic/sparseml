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
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import onnx
from onnx import ModelProto

from sparseml.base import Framework
from sparseml.benchmark import BatchBenchmarkResult, BenchmarkInfo, BenchmarkRunner
from sparseml.framework import FrameworkInfo
from sparseml.framework.info import FrameworkInferenceProviderInfo
from sparseml.onnx.base import require_onnx, require_onnxruntime
from sparseml.onnx.framework import framework_info as get_framework_info
from sparseml.onnx.framework import is_supported
from sparseml.onnx.utils import DataLoader, ORTModelRunner, max_available_cores
from sparsezoo import File, Model
from sparsezoo.utils import DataLoader as SparseZooDataLoader
from sparsezoo.utils import Dataset as SparseZooDataset


__all__ = [
    "ORTBenchmarkRunner",
    "load_model",
    "load_data",
    "run_benchmark",
]

_LOGGER = logging.getLogger(__name__)


GPU_ORT_PROVIDERS = [
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
]

CPU_DEFAULT_ORT_PROVIDER = "CPUExecutionProvider"


def _resolve_device_provider(
    framework_info: FrameworkInfo,
    device: Optional[str] = None,
    provider: Optional[str] = None,
) -> Tuple[str, str]:
    if provider is None and device is None:
        # Default to first available inference provider
        provider = framework_info.inference_providers[0].name
        device = framework_info.inference_providers[0].device
    elif provider is None:
        matching_provider = [
            inference_provider
            for inference_provider in framework_info.inference_providers
            if inference_provider.device == device
        ]
        if len(matching_provider) == 0:
            raise ValueError(f"No inference providers available for device {device}.")
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
    return device, provider


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
        ort_provider: Optional[str] = None,
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

        device, provider = _resolve_device_provider(
            self._framework_info, device=device, provider=provider
        )

        if "ort_provider" in framework_args:
            ort_provider = framework_args["ort_provider"]

        if ort_provider is None:
            if device == "cpu":
                ort_provider = CPU_DEFAULT_ORT_PROVIDER
            elif device == "gpu":
                possible_ort_providers = [
                    provider
                    for provider in GPU_ORT_PROVIDERS
                    if provider
                    in self._framework_info.properties["available_providers"]
                ]
                if len(possible_ort_providers) > 0:
                    ort_provider = possible_ort_providers[0]
                else:
                    _LOGGER.warn(
                        "No Onnx Runtime GPU providers installed. Defaulting to CPU"
                    )
                    device, provider = _resolve_device_provider(
                        self._framework_info, device="cpu"
                    )
                    ort_provider = CPU_DEFAULT_ORT_PROVIDER

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

    def run_batch(
        self, batch: Union[Dict[str, Any], Tuple[Dict[str, Any], Any]], *args, **kwargs
    ) -> BatchBenchmarkResult:
        """
        Runs a benchmark on a given batch.

        :param batch: the batch to benchmark
        :param args: additional arguments to pass to the framework
        :param kwargs: additional arguments to pass to the framework
        """
        # Handles case where batch consists of a tuple of input/labels
        if isinstance(batch, tuple):
            batch = batch[0]
        _, batch_time = self._model_runner.batch_forward(batch, *args, **kwargs)
        return BatchBenchmarkResult.from_result(batch_time, self.batch_size)

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
    def num_cores(self) -> str:
        """
        :return: the number of cores
        """
        return max_available_cores()

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

    @property
    def model(self) -> ModelProto:
        """
        :return: the model as an ONNX ModelProto
        """
        return self._model


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
        model = (
            Model(model, download_path=kwargs["path"])
            if "path" in kwargs
            else Model(model)
        )

    if isinstance(model, Model):
        # default to the main onnx file for the model
        model = model.onnx_model.path
    elif isinstance(model, File):
        # get the downloaded_path -- will auto download if not on local system
        model = model.path
    elif isinstance(model, ModelProto):
        return model

    if not isinstance(model, str):
        raise ValueError("unsupported type for model: {}".format(type(model)))

    if not os.path.exists(model):
        raise ValueError("model path must exist: given {}".format(model))

    return onnx.load(model)


def load_data(
    data: Any,
    model: Any = None,
    batch_size: int = 1,
    total_iterations: int = 0,
    **kwargs,
) -> Iterable[Tuple[Dict[str, Any], Any]]:
    """
    Creates a iteratable data loader for the given data.
    Acceptable types for data are:
    - a folder path containing numpy files
    - a list of file paths
    - a SparseML DataLoader
    - a SparseZoo DataLoader
    - an iterable
    - None type, in which case model must be passed
    :param data: data to use for benchmarking
    :param model: model to use for generating data
    :param batch_size: batch size
    :param total_iterations: total number of iterations
    :param kwargs: additional arguments to pass to the DataLoader
    :return: an iterable of data and labels
    """
    # Creates random data from model input shapes if data is not provided
    if not data:
        if not model:
            raise ValueError("must provide model or data")
        model = load_model(model)
        return DataLoader.from_model_random(
            model, batch_size, iter_steps=total_iterations
        )

    # If data is a SparseZoo stub, downloads model data
    if isinstance(data, str) and data.startswith("zoo:"):
        model_from_zoo = Model(data)
        data = model_from_zoo.sample_inputs.loader(
            batch_size, total_iterations, batch_as_list=False
        )

    # Imediately return the data if it is already a DataLoader
    if isinstance(data, DataLoader):
        return data

    # If data is a SparseZoo DataLoader, unbatches the dataloader and creates
    # DataLoader from it
    elif isinstance(data, SparseZooDataLoader):
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

    # If data is a dictionary of data shapes, creates DataLoader from random data
    elif isinstance(data, dict):
        is_dict_of_shapes = True
        for _, value in data.items():
            is_dict_of_shapes = is_dict_of_shapes and isinstance(value, tuple)
        if is_dict_of_shapes:
            return DataLoader.from_random(
                data,
                None,
                batch_size=batch_size,
                iter_steps=total_iterations,
                **kwargs,
            )

    # If data is a list of data shapes, creates DataLoader from random data
    elif isinstance(data, Iterable):
        element = next(iter(data))
        if isinstance(element, tuple):
            data_shapes = OrderedDict(
                (f"{index:04}", shape) for index, shape in enumerate(data)
            )
            return DataLoader.from_random(
                data_shapes,
                None,
                batch_size=batch_size,
                iter_steps=total_iterations,
                **kwargs,
            )
    return DataLoader(
        data, None, batch_size=batch_size, iter_steps=total_iterations, **kwargs
    )


@require_onnx()
@require_onnxruntime()
def run_benchmark(
    model: Any,
    data: Any = None,
    batch_size: int = 1,
    iterations: int = 0,
    warmup_iterations: int = 0,
    provider: Optional[str] = "cpu",
    device: Optional[str] = "cpu",
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
    model = load_model(model)

    if is_supported(model):
        benchmark_runner = ORTBenchmarkRunner(
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
            benchmark=results,
            config=benchmark_runner.benchmark_config,
        )
    else:
        raise ValueError(
            "Model is not supported by the onnxruntime backend. "
            "Please check the model for support."
        )
