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
Benchmarking PyTorch models on a given device for given batch sizes
"""

import time
from typing import Any, List, Tuple

import numpy
import torch
from torch.nn import Module
from tqdm import auto

from sparseml.pytorch.utils.helpers import (
    tensors_batch_size,
    tensors_module_forward,
    tensors_to_device,
    tensors_to_precision,
)
from sparseml.pytorch.utils.model import model_to_device


__all__ = ["BatchBenchmarkResults", "ModuleBenchmarker"]


class BatchBenchmarkResults(object):
    """
    Container class for the results of a benchmark run for a given batch size.
    Contains convenience methods for calculating different metrics around the time
    to run each batch and the items.

    :param batch_size: the batch size the results are for
    """

    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        self._batch_model_times = []
        self._batch_e2e_times = []

    def __repr__(self):
        return "{}(batch_size={}, batch_model_times={}, batch_e2e_times={})".format(
            self.__class__.__name__,
            self._batch_size,
            self._batch_model_times,
            self._batch_e2e_times,
        )

    def __str__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                [
                    "batch_size={}".format(self._batch_size),
                    "model_batch_seconds={}".format(self.model_batch_seconds),
                    "model_item_seconds={}".format(self.model_item_seconds),
                    "model_batches_per_second={}".format(self.model_batches_per_second),
                    "model_items_per_second={}".format(self.model_items_per_second),
                    "e2e_batch_seconds={}".format(self.e2e_batch_seconds),
                    "e2e_item_seconds={}".format(self.e2e_item_seconds),
                    "e2e_batches_per_second={}".format(self.e2e_batches_per_second),
                    "e2e_items_per_second={}".format(self.e2e_items_per_second),
                ]
            ),
        )

    @property
    def batch_size(self) -> int:
        """
        :return: the batch size the results are for
        """
        return self._batch_size

    @property
    def model_batch_timings(self) -> List[float]:
        """
        :return: the overall timings in seconds for each batch to run through the model.
            Does not include time for transferring data to and from device (if any)
        """
        return self._batch_model_times

    @property
    def e2e_batch_timings(self) -> List[float]:
        """
        :return: the overall timings in seconds for each batch to run through the model
            and the system.
            Includes model execution time as well as time to transfer the data to and
            from a device.
        """
        return self._batch_e2e_times

    @property
    def model_batch_seconds(self):
        """
        :return: the average time it took to execute the batches through the model.
            Does not include time for transferring data to and from device (if any)
        """
        return float(numpy.mean(self.model_batch_timings))

    @property
    def model_batches_per_second(self):
        """
        :return: inverse of model_batch_seconds
        """
        return 1.0 / self.model_batch_seconds

    @property
    def model_item_seconds(self):
        """
        :return: the batch averaged time it took in seconds to execute one item
            through the model (model_batch_seconds / batch_size).
            Does not include time for transferring data to and from device (if any)
        """
        return self.model_batch_seconds / self.batch_size

    @property
    def model_items_per_second(self):
        """
        :return: inverse of model_items_per_second
        """
        return 1.0 / self.model_item_seconds

    @property
    def e2e_batch_seconds(self):
        """
        :return: the average overall time to execute the batches through the model
            and the system.
            Includes model execution time as well as time to transfer the data to
            and from a device.
        """
        return float(numpy.mean(self.e2e_batch_timings))

    @property
    def e2e_batches_per_second(self):
        """
        :return: inverse of e2e_batch_seconds
        """
        return 1.0 / self.e2e_batch_seconds

    @property
    def e2e_item_seconds(self):
        """
        :return: the batch averaged overall time to execute the batches through the
            model and the system (e2e_batch_seconds / batch_size).
            Includes model execution time as well as time to transfer the data to
            and from a device.
        """
        return self.e2e_batch_seconds / self.batch_size

    @property
    def e2e_items_per_second(self):
        """
        :return: inverse of e2e_item_seconds
        """
        return 1.0 / self.e2e_item_seconds

    def add(self, model_sec: float, e2e_sec: float, batch_size: int):
        """
        Add a new batch result

        :param model_sec: the seconds it took to execute the model
        :param e2e_sec: the seconds it took to execute model and transfer to and
            from device
        :param batch_size: the size of the batch recorded
        """
        if batch_size != self._batch_size:
            raise ValueError(
                "batch_size of {} does not match the original batch_size {}".format(
                    batch_size, self._batch_size
                )
            )

        self._batch_model_times.append(model_sec)
        self._batch_e2e_times.append(e2e_sec)


class ModuleBenchmarker(object):
    """
    Convenience class for benchmarking a model on a given device for given batches
    at a given precision.

    :param module: the module to benchmark
    """

    def __init__(self, module: Module):
        self._module = module

    def run_batches_on_device(
        self,
        batches: List[Any],
        device: str,
        full_precision: bool = True,
        test_size: int = 100,
        warmup_size: int = 10,
    ) -> BatchBenchmarkResults:
        """
        :param batches: the batches to run through the model and benchmark,
            should all be of the same batch_size
        :param device: the device to run the model on, ex: cpu, cuda, cuda:0, cuda:0,1
        :param full_precision: True to run at float32, False to run at float16
        :param test_size: the number of batches to run and calculate timings over
        :param warmup_size: the number of batches to run before calculating timings
        :return: the batch results for benchmarking
        """
        module = self._module.eval()
        module = module.float() if full_precision else module.half()
        module, device, device_ids = model_to_device(module, device)
        default_device = (
            device
            if device_ids is None or len(device_ids) < 1
            else "{}:{}".format(device, device_ids[0])
        )
        batches = tensors_to_precision(batches, full_precision)

        def _infinite_batch_looper():
            while True:
                for batch in batches:
                    yield batch

        batch_iter = _infinite_batch_looper()

        if warmup_size > 0:
            for _ in auto.tqdm(
                range(warmup_size), desc="warming up...", total=warmup_size
            ):
                batch = next(batch_iter)
                ModuleBenchmarker._execute_batch_for_time(batch, module, default_device)

        batch_size = tensors_batch_size(next(batch_iter))
        results = BatchBenchmarkResults(batch_size)

        for _ in auto.tqdm(range(test_size), desc="testing...", total=test_size):
            batch = next(batch_iter)
            model_sec, e2e_sec, batch_size = ModuleBenchmarker._execute_batch_for_time(
                batch, module, device
            )
            results.add(model_sec, e2e_sec, batch_size)

        return results

    @staticmethod
    def _execute_batch_for_time(
        batch: Any, module: Module, device: str
    ) -> Tuple[float, float, int]:
        with torch.no_grad():
            batch = tensors_to_device(batch, "cpu")

            if "cuda" in device:
                torch.cuda.synchronize()

            e2e_start = time.time()
            x_tens = tensors_to_device(batch, device)

            if "cuda" in device:
                torch.cuda.synchronize()

            model_start = time.time()
            y_pred = tensors_module_forward(x_tens, module, check_feat_lab_inp=False)

            if "cuda" in device:
                torch.cuda.synchronize()

            model_end = time.time()
            y_pred_local = tensors_to_device(y_pred, "cpu")

            if "cuda" in device:
                torch.cuda.synchronize()

            e2e_end = time.time()
            batch_size = tensors_batch_size(batch)

            del x_tens
            del y_pred
            del y_pred_local

            return model_end - model_start, e2e_end - e2e_start, batch_size
