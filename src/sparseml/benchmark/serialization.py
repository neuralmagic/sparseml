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
Functionality related to serialization of the benchmarking run.
"""

from typing import Any, Dict, List, Optional, Union

import numpy
from pydantic import BaseModel, Field

from sparseml.base import Framework
from sparseml.framework import FrameworkInferenceProviderInfo


__all__ = [
    "BatchBenchmarkResult",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkInfo",
]


class BatchBenchmarkResult(BaseModel):
    """
    Class for storing the result of one batch of a benchmark run.

    Extends pydantic BaseModel class for serialization to and from json in addition
    to proper type checking on construction.
    """

    batch_time: float = Field(
        title="batch_time",
        description="Time to process a batch of data in seconds",
        gt=0.0,
    )
    batches_per_second: float = Field(
        title="batches_per_second",
        description="Number of batches processed per second",
        gt=0.0,
    )
    items_per_second: float = Field(
        title="items_per_second",
        description="Items processed per second",
        gt=0.0,
    )
    ms_per_batch: float = Field(
        title="ms_per_batch",
        description="Time to process a batch of data in milliseconds",
        gt=0.0,
    )
    ms_per_item: float = Field(
        title="ms_per_item",
        description="Time to process a single item in milliseconds",
        gt=0.0,
    )
    batch_size: int = Field(
        title="batch_size",
        description="Batch size of the result",
        ge=1,
    )

    @classmethod
    def from_result(cls, batch_time: float, batch_size: int) -> "BatchBenchmarkResult":
        """
        Creates a serialized BatchBenchmarkResult from a given batch time and
        batch size.

        :param batch_time: Time to process a batch of data in seconds
        :param batch_size: Batch size of the result
        :return: A serialized BatchBenchmarkResult
        """
        if batch_time <= 0:
            raise ValueError("batch_time must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batches_per_second = 1.0 / batch_time
        items_per_second = batch_size / batch_time
        ms_per_batch = batch_time * 1e3
        ms_per_item = batch_time * 1e3 / batch_size

        return BatchBenchmarkResult(
            batch_time=batch_time,
            batches_per_second=batches_per_second,
            items_per_second=items_per_second,
            ms_per_batch=ms_per_batch,
            ms_per_item=ms_per_item,
            batch_size=batch_size,
        )


class BenchmarkResult(BaseModel):
    """
    Class for storing the results of a benchmark run. Includes any statistics of the
    benchmark batch times.

    Extends pydantic BaseModel class for serialization to and from json in addition
    to proper type checking on construction.
    """

    results: List[BatchBenchmarkResult] = Field(
        default=[],
        title="results",
        description="The results of all benchmark runs",
    )
    batch_times_mean: float = Field(
        title="batch_times_mean",
        description="The mean time to complete a batch in seconds",
        ge=0.0,
    )
    batch_times_median: float = Field(
        title="batch_times_median",
        description="The median time to complete a batch in seconds",
        ge=0.0,
    )
    batch_times_std: float = Field(
        title="batch_times_std",
        description="The standard deviation of the time to complete a batch",
        ge=0.0,
    )
    batch_times_median_90: float = Field(
        title="batch_times_median_90",
        description="The median batch run times of the top 90% of the batch times",
        ge=0.0,
    )
    batch_times_median_95: float = Field(
        title="batch_times_median_95",
        description="The median batch run times of the top 95% of the batch times",
        ge=0.0,
    )
    batch_times_median_99: float = Field(
        title="batch_times_median_99",
        description="The median batch run times of the top 99% of the batch times",
        ge=0.0,
    )
    items_per_second: float = Field(
        title="items_per_second",
        description="The number of items processed per batch",
        ge=0.0,
    )
    batches_per_second: float = Field(
        title="batches_per_second",
        description="The number of batches processed per second",
        ge=0.0,
    )
    ms_per_batch: float = Field(
        title="ms_per_batch",
        description="The number of milliseconds per batch",
        ge=0.0,
    )
    ms_per_item: float = Field(
        title="ms_per_item",
        description="The number of milliseconds per item",
        ge=0.0,
    )
    num_items: int = Field(
        title="num_items",
        description="The number of items processed",
        ge=0,
    )
    num_batches: int = Field(
        title="num_batches",
        description="The number of batches processed",
        ge=0,
    )

    @classmethod
    def from_results(
        cls,
        results: List[Union[BatchBenchmarkResult, float]],
        batch_size: Optional[int] = None,
    ) -> "BenchmarkResult":
        """
        Creates a serialized BenchmarkResult from a given list of benchmark results.
        If the results are a list of batch times, the batch size is required.

        :param batch_time: Time to process a batch of data in seconds
        :param batch_size: Batch size of the result
        :return: A serialized BatchBenchmarkResult
        """
        if not results:
            raise ValueError("results must be non-empty")

        for index, result in enumerate(results):
            if not isinstance(result, BatchBenchmarkResult):
                results[index] = BatchBenchmarkResult.from_result(result, batch_size)
        batch_times = [r.batch_time for r in results]
        num_batches = len(batch_times)
        num_items = sum(r.batch_size for r in results)

        batch_times_mean = numpy.mean(batch_times).item()
        batch_times_median = numpy.median(batch_times).item()
        batch_times_std = numpy.std(batch_times).item()

        batches_per_second = num_batches / sum(batch_times)
        items_per_second = num_items / sum(batch_times)

        ms_per_batch = sum(batch_times) * 1e3 / num_batches
        ms_per_item = sum(batch_times) * 1e3 / num_items

        sorted_batch_times = sorted(batch_times)
        top_90_index = int((num_batches) * 0.9)
        top_95_index = int((num_batches) * 0.95)
        top_99_index = int((num_batches) * 0.99)

        batch_times_median_90 = (
            numpy.median(sorted_batch_times[:top_90_index]).item()
            if top_90_index > 1
            else sorted_batch_times[0]
        )
        batch_times_median_95 = (
            numpy.median(sorted_batch_times[:top_95_index]).item()
            if top_95_index > 1
            else sorted_batch_times[0]
        )
        batch_times_median_99 = (
            numpy.median(sorted_batch_times[:top_99_index]).item()
            if top_99_index > 1
            else sorted_batch_times[0]
        )

        return BenchmarkResult(
            results=results,
            batch_times_mean=batch_times_mean,
            batch_times_median=batch_times_median,
            batch_times_std=batch_times_std,
            batch_times_median_90=batch_times_median_90,
            batch_times_median_95=batch_times_median_95,
            batch_times_median_99=batch_times_median_99,
            items_per_second=items_per_second,
            batches_per_second=batches_per_second,
            ms_per_batch=ms_per_batch,
            ms_per_item=ms_per_item,
            num_items=num_items,
            num_batches=num_batches,
        )


class BenchmarkConfig(BaseModel):
    """
    Class for storing the configuration of the benchmark run.

    Extends pydantic BaseModel class for serialization to and from json in addition
    to proper type checking on construction.
    """

    batch_size: int = Field(
        default=1,
        ge=1,
        title="batch_size",
        description="The batch size used for benchmarking",
    )
    iterations: int = Field(
        default=1,
        ge=0,
        title="iterations",
        description="The number of iteration steps used for benchmarking",
    )
    warmup_iterations: int = Field(
        default=0,
        ge=0,
        title="warmup_iterations",
        description="The number of warmup iterations used for benchmarking",
    )
    num_cores: int = Field(
        default=1,
        ge=1,
        title="num_cores",
        description="The number of cores used for benchmarking",
    )
    device: str = Field(
        title="device",
        description="The device the framework is running on.",
    )
    framework_args: Dict[str, Any] = Field(
        default={},
        title="framework_args",
        description="The framework specific arguments passed to the framework",
    )
    inference_provider: FrameworkInferenceProviderInfo = Field(
        title="inference_provides",
        description=("The inference provider info for the framework."),
    )


class BenchmarkInfo(BaseModel):
    """
    Class for storing the information of the benchmark run. Include configurations,
    results, and package versioning.

    Extends pydantic BaseModel class for serialization to and from json in addition
    to proper type checking on construction.
    """

    framework: Framework = Field(
        title="framework", description="The framework the system info is for."
    )
    package_versions: Dict[str, Optional[str]] = Field(
        title="package_versions",
        description=(
            "A mapping of the package and supporting packages for a given framework "
            "to the detected versions on the system currently. "
            "If the package is not detected, will be set to None."
        ),
    )
    benchmark: BenchmarkResult = Field(
        title="benchmark",
        description="The benchmark results for the framework.",
    )
    config: BenchmarkConfig = Field(
        title="config",
        description="The benchmark configuration for running the benchmark.",
    )
