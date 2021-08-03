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
Code related to benchmarking batched inference runs
"""

from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import numpy

from sparseml.benchmark.serialization import (
    BatchBenchmarkResultSchema,
    BenchmarkResultSchema,
)


__all__ = ["BatchBenchmarkResult", "BenchmarkResults"]


class BatchBenchmarkResult(object):
    """
    A benchmark result for a batched inference run

    :param batch_time: The time taken to process a batch
    :param batch_size: The size of the batch that was benchmarked
    :param inputs: Optional batch inputs that were given for the run
    :param outputs: Optional batch outputs that were given for the run
    :param extras: Optional batch extras to store any other data for the run
    """

    def __init__(
        self,
        batch_time: float,
        batch_size: int,
        inputs: Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]] = None,
        outputs: Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]] = None,
        extras: Any = None,
    ):
        if batch_time <= 0:
            raise ValueError("batch_time must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._batch_time = batch_time
        self._batch_size = batch_size
        self._inputs = inputs
        self._outputs = outputs
        self._extras = extras

    def __repr__(self):
        props = {
            "batch_time": self.batch_time,
            "size": self.batch_size,
            "batches_per_second": self.batches_per_second,
            "items_per_second": self.items_per_second,
            "ms_per_batch": self.ms_per_batch,
            "ms_per_item": self.ms_per_item,
        }

        return f"{self.__class__.__name__}({props})"

    def __str__(self):
        return (
            f"{self.__class__.__name__}(ms_per_batch={self.ms_per_batch}, "
            f"items_per_second={self.items_per_second})"
        )

    @property
    def batch_time(self) -> float:
        """
        :return: The time elapsed for the entire run (end - start)
        """
        return self._batch_time

    @property
    def batch_size(self) -> int:
        """
        :return: The size of the batch that was benchmarked
        """
        return self._batch_size

    @property
    def inputs(self) -> Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]]:
        """
        :return: Batch inputs that were given for the run, if any
        """
        return self._inputs

    @property
    def outputs(self) -> Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]]:
        """
        :return: Batch outputs that were given for the run, if any
        """
        return self._outputs

    @property
    def extras(self) -> Any:
        """
        :return: Batch extras to store any other data for the run
        """
        return self._extras

    @property
    def batches_per_second(self) -> float:
        """
        :return: The number of batches that could be run in one second
            based on this result
        """
        return 1.0 / self.batch_time

    @property
    def items_per_second(self) -> float:
        """
        :return: The number of items that could be run in one second
            based on this result
        """
        return self._batch_size / self.batch_time

    @property
    def ms_per_batch(self) -> float:
        """
        :return: The number of milliseconds it took to run the batch
        """
        return self.batch_time * 1e3

    @property
    def ms_per_item(self) -> float:
        """
        :return: The averaged number of milliseconds it took to run each item
            in the batch
        """
        return self.batch_time * 1e3 / self._batch_size

    def dict(self) -> Dict[str, Any]:
        """
        :return: The dict representation of this object
        """
        return BatchBenchmarkResultSchema(
            batch_time=self.batch_time,
            batch_size=self.batch_size,
            batches_per_second=self.batches_per_second,
            items_per_second=self.items_per_second,
            ms_per_batch=self.ms_per_batch,
            ms_per_item=self.ms_per_item,
        ).dict()


class BenchmarkResults(Iterable):
    """
    The benchmark results for a list of batched inference runs
    """

    def __init__(self):
        self._results: List[BatchBenchmarkResult] = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self._properties_dict})"

    def __str__(self):
        """
        :return: Human readable form of the benchmark summary
        """
        formatted_props = [
            "\t{}: {}".format(key, val) for key, val in self._properties_dict.items()
        ]
        return "{}:\n{}".format(
            self.__class__.__name__,
            "\n".join(formatted_props),
        )

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, index: int) -> BatchBenchmarkResult:
        return self._results[index]

    def __iter__(self) -> Iterator[BatchBenchmarkResult]:
        for result in self._results:
            yield result

    @property
    def _properties_dict(self) -> Dict:
        return {
            "items_per_second": self.items_per_second,
            "ms_per_batch": self.ms_per_batch,
            "batch_times_mean": self.batch_times_mean,
            "batch_times_median": self.batch_times_median,
            "batch_times_std": self.batch_times_std,
        }

    @property
    def results(self) -> List[BatchBenchmarkResult]:
        """
        :return: the list of recorded batch results
        """
        return self._results

    @property
    def num_batches(self) -> int:
        """
        :return: the number of batches that have been added
        """
        return len(self)

    @property
    def num_items(self) -> int:
        """
        :return: the number of items across all batches that have been added
        """
        num_items = sum([res.batch_size for res in self._results])

        return num_items

    @property
    def batch_times(self) -> List[float]:
        """
        :return: the list of all batch run times that have been added
        """
        return [res.batch_time for res in self._results]

    @property
    def batch_sizes(self) -> List[int]:
        """
        :return: the list of all batch run sizes that have been added
        """
        return [res.batch_size for res in self._results]

    @property
    def batch_times_mean(self) -> float:
        """
        :return: the mean of all the batch run times that have been added
        """
        if len(self) == 0:
            return 0.0
        return numpy.mean(self.batch_times).item()

    @property
    def batch_times_median(self) -> float:
        """
        :return: the median of all the batch run times that have been added
        """
        if len(self) == 0:
            return 0.0
        return numpy.median(self.batch_times).item()

    @property
    def batch_times_std(self) -> float:
        """
        :return: the standard deviation of all the batch run times that have been added
        """
        if len(self) == 0:
            return 0.0
        return numpy.std(self.batch_times).item()

    @property
    def batches_per_second(self) -> float:
        """
        :return: The number of batches that could be run in one second
            based on this result
        """
        if len(self) == 0:
            return 0.0
        return self.num_batches / sum(self.batch_times)

    @property
    def items_per_second(self) -> float:
        """
        :return: The number of items that could be run in one second
            based on this result
        """
        if len(self) == 0:
            return 0.0
        return self.num_items / sum(self.batch_times)

    @property
    def ms_per_batch(self) -> float:
        """
        :return: The number of milliseconds it took to run the batch
        """
        if len(self) == 0:
            return 0.0
        return sum(self.batch_times) * 1000.0 / self.num_batches

    @property
    def ms_per_item(self) -> float:
        """
        :return: The averaged number of milliseconds it took to run each item
            in the batch
        """
        if len(self) == 0:
            return 0.0
        return sum(self.batch_times) * 1000.0 / self.num_items

    @property
    def batch_times_median_90(self) -> float:
        """
        :return: The median batch run times of the top 90% of the batch times
        """
        if len(self) == 0:
            return 0.0
        if len(self) == 1:
            return self.batch_times_median
        sorted_batch_times = numpy.sort(self.batch_times)
        top_90_index = int(len(sorted_batch_times) * 0.9)
        return numpy.median(sorted_batch_times[:top_90_index]).item()

    @property
    def batch_times_median_95(self) -> float:
        """
        :return: The median batch run times of the top 95% of the batch times
        """
        if len(self) == 0:
            return 0.0
        if len(self) == 1:
            return self.batch_times_median
        sorted_batch_times = numpy.sort(self.batch_times)
        top_95_index = int(len(sorted_batch_times) * 0.95)
        return numpy.median(sorted_batch_times[:top_95_index]).item()

    @property
    def batch_times_median_99(self) -> float:
        """
        :return: The median batch run times of the top 99% of the batch times
        """
        if len(self) == 0:
            return 0.0
        if len(self) == 1:
            return self.batch_times_median
        sorted_batch_times = numpy.sort(self.batch_times)
        top_99_index = int(len(sorted_batch_times) * 0.99)
        return numpy.median(sorted_batch_times[:top_99_index]).item()

    @property
    def inputs(
        self,
    ) -> List[Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]]]:
        """
        :return: Batch inputs that were given for the run, if any
        """
        return [res.inputs for res in self._results]

    @property
    def outputs(
        self,
    ) -> List[Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]]]:
        """
        :return: Batch outputs that were given for the run, if any
        """
        return [res.outputs for res in self._results]

    def append_batch(
        self,
        batch_time: float,
        batch_size: int,
        inputs: Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]] = None,
        outputs: Optional[Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]] = None,
        extras: Any = None,
    ):
        """
        Add a recorded batch to the current results

        :param batch_time: The time it took to run the batch
        :param batch_size: The size of the batch that was benchmarked
        :param inputs: Optional batch inputs that were given for the run
        :param outputs: Optional batch outputs that were given for the run
        :param extras: Optional batch extras to store any other data for the run
        """
        self._results.append(
            BatchBenchmarkResult(batch_time, batch_size, inputs, outputs, extras)
        )

    def append(
        self,
        batch_result: BatchBenchmarkResult,
    ):
        """
        Add a batch result to the current results

        :param batch_result: The batch result to add
        """
        self._results.append(batch_result)

    def dict(self) -> Dict[str, Any]:
        """
        :return: The dict representation of this object
        """
        return BenchmarkResultSchema(
            results=[result.dict() for result in self._results],
            batch_times_mean=self.batch_times_mean,
            batch_times_median=self.batch_times_median,
            batch_times_std=self.batch_times_std,
            batch_times_median_90=self.batch_times_median_90,
            batch_times_median_95=self.batch_times_median_95,
            batch_times_median_99=self.batch_times_median_99,
            batches_per_second=self.batches_per_second,
            items_per_second=self.items_per_second,
            ms_per_batch=self.ms_per_batch,
            ms_per_item=self.ms_per_item,
            num_items=self.num_items,
            num_batches=self.num_batches,
        ).dict()
