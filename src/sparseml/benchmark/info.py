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

"""
Functionality related to running a benchmark for a given model.

The file is executable and will run a benchmark for the specified model.

##########
Command help
usage: sparseml.benchmark [-h] --model MODEL [--data DATA]
                          [--batch-size BATCH_SIZE] [--iterations ITERATIONS]
                          [--warmup-iterations WARMUP_ITERATIONS]
                          [--framework FRAMEWORK] [--provider PROVIDER]
                          [--device DEVICE] [--save-path SAVE_PATH]
                          [--show-progress]

Run a benchmark for a specific model in an optionally specified framework.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The model used for inference. Accepts either a path to
                        the directory where the model is saved or a zoo stub.
  --data DATA           The path to the directory where the data is saved.
  --batch-size BATCH_SIZE
                        The batch size to use for the benchmark. If not
                        specified, will be set to 1.
  --iterations ITERATIONS
                        The number of iteration steps to use for the
                        benchmark. If not specified, will be set to 0 and go
                        through entire dataset once.
  --warmup-iterations WARMUP_ITERATIONS
                        The number of warmup iterations to use for the
                        benchmark.
  --framework FRAMEWORK
                        The framework to use for the benchmark. If not
                        specified, will be automatically detected based on
                        model provided.
  --provider PROVIDER   The inference provider to use for the benchmark. If
                        not specified, will be automatically detected.
  --device DEVICE       The device to use for the benchmark. If not specified,
                        will be automatically detected.
  --save-path SAVE_PATH
                        A full file path to save the benchmark results to. If
                        not supplied, will print out the benchmark results to
                        the console.
  --show-progress       If specified, will show the progress of the benchmark.

#########
EXAMPLES
#########

##########
Example command for running a benchmark with onnxruntime for a model via SparseZoo stub

sparseml.benchmark --framework onnx \
    --model zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none

##########
Example command for running a benchmark with onnxruntime for a model via SparseZoo stub
with specified batch size, warmup iterations and iterations

sparseml.benchmark --framework onnx --batch-size 32 \
    --iterations 100 --warmup-iterations 10 \
    --model zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none

##########
Example command for running a benchmark with onnxruntime for a model via local path

sparseml.benchmark --framework onnx --model ~/downloads/model.onnx

##########
Example command for running a benchmark with onnxruntime for a model via local path and
with a specific data path

sparseml.benchmark --framework onnx --model ~/downloads/model.onnx \
    --data ~/downloads/sample-inputs
"""

import argparse
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, Optional

from tqdm import auto

from sparseml.base import Framework, execute_in_sparseml_framework
from sparseml.benchmark.serialization import (
    BatchBenchmarkResult,
    BenchmarkConfig,
    BenchmarkInfo,
    BenchmarkResult,
)
from sparseml.framework.info import FrameworkInferenceProviderInfo, FrameworkInfo
from sparseml.utils import clean_path, create_parent_dirs
from sparseml.utils.helpers import convert_to_bool


__all__ = [
    "BenchmarkRunner",
    "save_benchmark_results",
    "load_benchmark_info",
    "load_and_run_benchmark",
]


_LOGGER = logging.getLogger(__name__)


class BenchmarkRunner(ABC):
    """
    Abstract class for handling running benchmarks with different frameworks.
    """

    def run(
        self,
        data: Any,
        desc: str = "",
        load_data_kwargs: Dict[str, Any] = {},
        show_progress: bool = False,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Runs a benchmark on the given data. Results are serialized together using
        the BenchmarkResult class.

        :param data: data to use for benchmarking
        :param desc: str to display if show_progress is True
        :param show_progress: whether to show progress
        :param load_data_kwargs: additional arguments to pass to the framework's
            load_data method
        :param args: additional arguments to pass to the framework
        :param kwargs: additional arguments to pass to the framework
        :return: the results of the benchmark run
        :rtype: BenchmarkResult
        """
        results = []
        for batch_result in self.run_iter(
            data,
            desc=desc,
            show_progress=show_progress,
            load_data_kwargs=load_data_kwargs,
            *args,
            **kwargs,
        ):
            results.append(batch_result)
        return BenchmarkResult.from_results(results)

    def run_iter(
        self,
        data: Any,
        desc: str = "",
        show_progress: bool = False,
        load_data_kwargs: Dict[str, Any] = {},
        *args,
        **kwargs,
    ) -> Iterator[BatchBenchmarkResult]:
        """
        Iteratively runs a benchmark on the given data. Non warmup iterations
        results are returned serialized as BatchBenchmarkResult.

        :param data: data to use for benchmarking
        :param desc: str to display if show_progress is True
        :param show_progress: whether to show progress
        :param load_data_kwargs: additional arguments to pass to the framework's
            load_data method
        :param args: additional arguments to pass to the framework
        :param kwargs: additional arguments to pass to the framework
        :return: an iterator of the benchmark results for each batch
        :rtype: Iterator[BatchBenchmarkResult]
        """
        _LOGGER.debug("loading data with load_model")
        loaded_data = self.load_data(data, **load_data_kwargs)

        progress_steps = self.warmup_iterations + self.iterations
        _LOGGER.debug("running {} items through model".format(progress_steps))
        data_iter = (
            enumerate(loaded_data)
            if not show_progress
            else auto.tqdm(enumerate(loaded_data), desc=desc, total=progress_steps)
        )
        for index, batch in data_iter:
            if index < self.warmup_iterations:
                self.run_batch(batch, *args, **kwargs)
                continue
            yield self.run_batch(batch, *args, **kwargs)

    def load_data(self, data: Any, **kwargs) -> Iterable[Any]:
        """
        Uses the framework's load_data method to load the data into
        an iterable for use in benchmarking.

        :param data: data to load
        :param kwargs: additional arguments to pass to the framework's load_data method
        :return: an iterable of the loaded data
        """
        return execute_in_sparseml_framework(
            self.framework,
            "load_data",
            data=data,
            model=self.model,
            batch_size=self.batch_size,
            total_iterations=self.warmup_iterations + self.iterations,
            **kwargs,
        )

    @abstractmethod
    def run_batch(
        self,
        batch: Any,
        *args,
        **kwargs,
    ) -> BatchBenchmarkResult:
        """
        Runs a benchmark on a given batch.

        :param batch: the batch to benchmark
        :param args: additional arguments to pass to the framework
        :param kwargs: additional arguments to pass to the framework
        """
        raise NotImplementedError()

    @property
    def benchmark_config(self) -> BenchmarkConfig:
        """
        :return: The benchmark configuration.
        """
        return BenchmarkConfig(
            batch_size=self.batch_size,
            iterations=self.iterations,
            warmup_iterations=self.warmup_iterations,
            device=self.device,
            inference_provider=self.inference_provider,
            num_cores=self.num_cores,
        )

    @property
    @abstractmethod
    def framework(self) -> Framework:
        """
        :return: the framework
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def framework_info(self) -> FrameworkInfo:
        """
        :return: the framework info
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        :return: the batch size
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def warmup_iterations(self) -> int:
        """
        :return: the number of warmup iterations
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def iterations(self) -> int:
        """
        :return: the number of iterations
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_cores(self) -> int:
        """
        :return: the number of cores
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def framework_args(self) -> Dict[str, Any]:
        """
        :return: the framework args
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def inference_provider(self) -> FrameworkInferenceProviderInfo:
        """
        :return: the inference provider
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def package_versions(self) -> Dict[str, str]:
        """
        :return: the package versions
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def device(self) -> Optional[str]:
        """
        :return: the device
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def model(self) -> Any:
        """
        :return: the model as ran in the benchmark
        """
        raise NotImplementedError()


def save_benchmark_results(
    model: Any,
    data: Any,
    batch_size: int,
    iterations: int,
    warmup_iterations: int,
    framework: Optional[str],
    provider: Optional[str] = None,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    framework_args: Dict[str, Any] = {},
    show_progress: bool = False,
):
    """
    Saves the benchmark results ran for specific framework.
    If path is provided, will save to a json file at the path.
    If path is not provided, will print out the info.

    If no framework is provided, will detect the framework based on the model.

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
    """
    results = execute_in_sparseml_framework(
        framework if framework is not None else model,
        "run_benchmark",
        model,
        data,
        batch_size=batch_size,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        provider=provider,
        device=device,
        framework_args=framework_args,
        show_progress=show_progress,
    )

    if save_path:
        save_path = clean_path(save_path)
        create_parent_dirs(save_path)

        with open(save_path, "w") as file:
            file.write(results.json(indent=4))

        _LOGGER.info(f"saved benchmark results in file at {save_path}"),
    else:
        print(results.json(indent=4))
        _LOGGER.info("printed out benchmark results")


def load_benchmark_info(load: str) -> BenchmarkInfo:
    """
    Load the benchmark info from a file or raw json.
    If load exists as a path, will read from the file and use that.
    Otherwise will try to parse the input as a raw json str.

    :param load: Either a file path to a json file or a raw json string.
    :type load: str
    :return: The loaded benchmark info.
    :rtype: FrameworkInfo
    """
    loaded_path = clean_path(load)

    if os.path.exists(loaded_path):
        with open(loaded_path, "r") as file:
            load = file.read()

    info = BenchmarkInfo.parse_raw(load)

    return info


def load_and_run_benchmark(
    model: Any,
    data: Any,
    load: str,
    save_path: Optional[str] = None,
):
    """
    Loads the benchmark configuration from a file or raw json and reruns
    the benchmark.

    If load exists as a path, will read from the file and use that.
    Otherwise will try to parse the input as a raw json str.

    :param model: model to benchmark
    :param data: data to benchmark
    :param load: Either a file path to a json file or a raw json string.
    :param save_path: path to save the new benchmark results
    """
    _LOGGER.info(f"rerunning benchmark {load}")
    info = load_benchmark_info(load)
    save_benchmark_results(
        info.framework if info.framework is not None else model,
        data,
        batch_size=info.config.batch_size,
        iterations=info.config.iterations,
        warmup_iterations=info.config.warmup_iterations,
        framework=info.framework,
        provider=info.config.inference_provider.name,
        device=info.config.device,
        framework_args=info.config.framework_args,
        save_path=save_path,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a benchmark for a specific model in an optionally specified framework."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "The model used for inference. Accepts either a path to the directory "
            "where the model is saved or a zoo stub."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help=("The path to the directory where the data is saved."),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "The batch size to use for the benchmark. If not specified, will "
            "be set to 1."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help=(
            "The number of iteration steps to use for the benchmark. If not specified, "
            "will be set to 0 and go through entire dataset once."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=0,
        help=("The number of warmup iterations to use for the benchmark."),
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help=(
            "The framework to use for the benchmark. If not specified, will be "
            "automatically detected based on model provided."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help=(
            "The inference provider to use for the benchmark. If not specified, will "
            "be automatically detected."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "The device to use for the benchmark. If not specified, will be "
            "automatically detected."
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=(
            "A full file path to save the benchmark results to. "
            "If not supplied, will print out the benchmark results to the console."
        ),
    )
    parser.add_argument(
        "--show-progress",
        type=convert_to_bool,
        default=True,
        help=("If specified, will show the progress of the benchmark."),
    )

    return parser.parse_args()


def _main():
    args = _parse_args()
    save_benchmark_results(
        model=args.model,
        data=args.data,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        framework=args.framework,
        provider=args.provider,
        device=args.device,
        save_path=args.save_path,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    _main()
