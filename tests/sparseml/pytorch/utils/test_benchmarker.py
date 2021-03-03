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

import os
import sys

import numpy
import pytest
import torch

from sparseml.pytorch.utils import BatchBenchmarkResults, ModuleBenchmarker
from tests.sparseml.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_results_const():
    batch_size = 1
    results = BatchBenchmarkResults(batch_size)
    assert results.batch_size == batch_size
    assert len(results.model_batch_timings) < 1
    assert len(results.e2e_batch_timings) < 1


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_results_add():
    batch_size = 1
    results = BatchBenchmarkResults(batch_size)

    results.add(1.0, 1.0, 1)
    assert len(results.model_batch_timings) == 1
    assert len(results.e2e_batch_timings) == 1

    with pytest.raises(ValueError):
        results.add(1.0, 1.0, 8)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize(
    "timings,avg",
    [([0.1, 0.1, 0.1, 0.1], 0.1), ([0.0, 1.0, 2.0], 1.0), ([0.0, -1.0, -2.0], -1.0)],
)
def test_results_props_single_batch_size(batch_size, timings, avg):
    results = BatchBenchmarkResults(batch_size)
    for val in timings:
        results.add(val, 0.0, batch_size)
    assert len(results.model_batch_timings) == len(timings)
    assert (
        abs(results.model_batch_seconds - numpy.average(timings))
        < sys.float_info.epsilon
    )
    assert (
        abs(results.model_batches_per_second - 1.0 / numpy.average(timings))
        < sys.float_info.epsilon
    )
    assert (
        abs(results.model_item_seconds - numpy.average(timings) / batch_size)
        < sys.float_info.epsilon
    )
    assert (
        abs(results.model_items_per_second - 1.0 / numpy.average(timings) * batch_size)
        < sys.float_info.epsilon
    )
    assert results.e2e_batch_seconds == 0.0
    assert results.e2e_batch_seconds == 0.0

    results = BatchBenchmarkResults(batch_size)
    for val in timings:
        results.add(0.0, val, batch_size)
    assert len(results.e2e_batch_timings) == len(timings)
    assert (
        abs(results.e2e_batch_seconds - numpy.average(timings)) < sys.float_info.epsilon
    )
    assert (
        abs(results.e2e_batches_per_second - 1.0 / numpy.average(timings))
        < sys.float_info.epsilon
    )
    assert (
        abs(results.e2e_item_seconds - numpy.average(timings) / batch_size)
        < sys.float_info.epsilon
    )
    assert (
        abs(results.e2e_items_per_second - 1.0 / numpy.average(timings) * batch_size)
        < sys.float_info.epsilon
    )
    assert results.model_batch_seconds == 0.0
    assert results.model_batch_seconds == 0.0


def _results_sanity_check(
    results: BatchBenchmarkResults, test_size: int, batch_size: int
):
    assert len(results.model_batch_timings) == test_size
    assert len(results.e2e_batch_timings) == test_size
    assert results.batch_size == batch_size

    assert results.model_batch_seconds > 0.0
    assert results.model_batches_per_second > 0.0
    assert results.model_item_seconds > 0.0
    assert results.model_items_per_second > 0.0
    assert results.e2e_batch_seconds > 0.0
    assert results.e2e_batches_per_second > 0.0
    assert results.e2e_item_seconds > 0.0
    assert results.e2e_items_per_second > 0.0


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("batch_size", [1, 64])
def test_benchmark_cpu(batch_size):
    benchmarker = ModuleBenchmarker(MLPNet())
    batches = [torch.rand(batch_size, 8) for _ in range(10)]
    warmup_size = 5
    test_size = 30

    results = benchmarker.run_batches_on_device(
        batches,
        "cpu",
        full_precision=True,
        warmup_size=warmup_size,
        test_size=test_size,
    )
    _results_sanity_check(results, test_size, batch_size)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_benchmark_cuda_full(batch_size):
    benchmarker = ModuleBenchmarker(MLPNet())
    batches = [torch.rand(batch_size, 8) for _ in range(10)]
    warmup_size = 5
    test_size = 30

    results = benchmarker.run_batches_on_device(
        batches,
        "cuda",
        full_precision=True,
        warmup_size=warmup_size,
        test_size=test_size,
    )
    _results_sanity_check(results, test_size, batch_size)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_benchmark_cuda(batch_size):
    benchmarker = ModuleBenchmarker(MLPNet())
    batches = [torch.rand(batch_size, 8) for _ in range(10)]
    warmup_size = 5
    test_size = 30

    results = benchmarker.run_batches_on_device(
        batches,
        "cuda",
        full_precision=False,
        warmup_size=warmup_size,
        test_size=test_size,
    )
    _results_sanity_check(results, test_size, batch_size)
