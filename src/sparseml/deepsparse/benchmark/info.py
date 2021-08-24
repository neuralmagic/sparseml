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
from typing import Any, Dict, Optional, Tuple, Union

from onnx import ModelProto

from sparseml.base import Framework
from sparseml.benchmark import BatchBenchmarkResult, BenchmarkInfo, BenchmarkRunner
from sparseml.deepsparse.base import require_deepsparse
from sparseml.deepsparse.framework import framework_info as get_framework_info
from sparseml.deepsparse.framework import is_supported
from sparseml.framework import FrameworkInfo
from sparseml.framework.info import FrameworkInferenceProviderInfo
from sparseml.onnx.benchmark.info import load_data, load_model
from sparseml.onnx.utils import DeepSparseModelRunner


__all__ = [
    "DeepSparseBenchmarkRunner",
    "load_model",
    "load_data",
    "run_benchmark",
]

_LOGGER = logging.getLogger(__name__)


class DeepSparseBenchmarkRunner(BenchmarkRunner):
    PROVIDER = "cpu"
    DEVICE = "cpu"

    @require_deepsparse()
    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        iterations: int = 0,
        warmup_iterations: int = 0,
        num_cores: Optional[int] = None,
        num_sockets: Optional[int] = None,
        framework_args: Dict[str, Any] = {},
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

        if "num_cores" in framework_args and num_cores is not None:
            framework_args["num_cores"] = num_cores
            _LOGGER.warn(
                f'Provided "num_cores" {num_cores} overriding "num_cores" '
                "field in framework_args"
            )

        if "num_sockets" in framework_args and num_sockets is not None:
            framework_args["num_sockets"] = num_sockets
            _LOGGER.warn(
                f'Provided "num_sockets" {num_sockets} overriding "num_sockets" '
                "field in framework_args"
            )

        if not framework_args:
            framework_args = {
                "num_cores": None,
                "num_sockets": None,
            }

        self._model = load_model(model)

        self._framework_info = get_framework_info()
        self._package_versions = self._framework_info.package_versions
        inference_providers = [
            inference_provider
            for inference_provider in self._framework_info.inference_providers
            if inference_provider.name == DeepSparseBenchmarkRunner.PROVIDER
            and inference_provider.device == DeepSparseBenchmarkRunner.DEVICE
        ]
        if len(inference_providers) == 0:
            raise ValueError(
                "No supported inference provider found for "
                f"{DeepSparseBenchmarkRunner.PROVIDER}."
            )

        self._model_runner = DeepSparseModelRunner(
            self._model,
            batch_size=batch_size,
            # num_cores=num_cores,
            # num_sockets=num_sockets,
            **framework_args,
        )

        self._inference_provider = inference_providers[0]
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
        return Framework.deepsparse

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
        return self.DEVICE

    @property
    def model(self) -> ModelProto:
        """
        :return: the model as an ONNX ModelProto
        """
        return self._model


@require_deepsparse()
def run_benchmark(
    model: Any,
    data: Any = None,
    batch_size: int = 1,
    iterations: int = 0,
    warmup_iterations: int = 0,
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
    if is_supported(model):
        benchmark_runner = DeepSparseBenchmarkRunner(
            model,
            batch_size=batch_size,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            framework_args=framework_args,
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
            "Model is not supported by the deepsparse backend. "
            "Please check the model for support."
        )
