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

import random

import numpy
import pytest

from sparseml.benchmark.results import BatchBenchmarkResult, BenchmarkResults


class TestBatchBenchmarkResult:
    @pytest.mark.parametrize(
        "inputs",
        [
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                    inputs=None,
                    outputs=None,
                    extras=None,
                )
            ),
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                    inputs=[numpy.random.rand(5, 5)],
                    outputs=[numpy.random.rand(5, 5)],
                    extras=[numpy.random.rand(5, 5)],
                )
            ),
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                    inputs=[
                        dict(in1=numpy.random.rand(5, 5), in2=numpy.random.rand(5, 5))
                    ],
                    outputs=[
                        dict(out1=numpy.random.rand(5, 5), out2=numpy.random.rand(5, 5))
                    ],
                    extras=[numpy.random.rand(5, 5)],
                )
            ),
        ],
    )
    def test_init(self, inputs):
        result = BatchBenchmarkResult(**inputs)
        for key in inputs:
            assert getattr(result, key) == inputs[key]

    @pytest.mark.parametrize(
        "inputs",
        [
            (
                dict(
                    batch_time=0.0,
                    batch_size=1,
                )
            ),
            (
                dict(
                    batch_time=1.0,
                    batch_size=0,
                )
            ),
        ],
    )
    def test_init_failures(self, inputs):
        with pytest.raises(ValueError):
            BatchBenchmarkResult(**inputs)

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                ),
                1.0,
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=1,
                ),
                0.5,
            ),
            (
                dict(
                    batch_time=3.0,
                    batch_size=1,
                ),
                1 / 3,
            ),
        ],
    )
    def test_batches_per_second(self, inputs, expected_output):
        result = BatchBenchmarkResult(**inputs)
        assert result.batches_per_second == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                ),
                1.0,
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=1,
                ),
                0.5,
            ),
            (
                dict(
                    batch_time=3.0,
                    batch_size=1,
                ),
                1 / 3,
            ),
            (
                dict(
                    batch_time=1.0,
                    batch_size=2,
                ),
                2.0,
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=10,
                ),
                5.0,
            ),
        ],
    )
    def test_items_per_second(self, inputs, expected_output):
        result = BatchBenchmarkResult(**inputs)
        assert result.items_per_second == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                ),
                1.0 * 1e3,
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=1,
                ),
                2 * 1e3,
            ),
            (
                dict(
                    batch_time=3.0,
                    batch_size=1,
                ),
                3 * 1e3,
            ),
        ],
    )
    def test_ms_per_batch(self, inputs, expected_output):
        result = BatchBenchmarkResult(**inputs)
        assert result.ms_per_batch == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                ),
                1.0 * 1e3,
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=1,
                ),
                2.0 * 1e3,
            ),
            (
                dict(
                    batch_time=3.0,
                    batch_size=1,
                ),
                3.0 * 1e3,
            ),
            (
                dict(
                    batch_time=1.0,
                    batch_size=2,
                ),
                0.5 * 1e3,
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=10,
                ),
                0.2 * 1e3,
            ),
        ],
    )
    def test_ms_per_item(self, inputs, expected_output):
        result = BatchBenchmarkResult(**inputs)
        assert result.ms_per_item == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                dict(
                    batch_time=1.0,
                    batch_size=1,
                ),
                dict(
                    batch_time=1.0,
                    batch_size=1,
                    batches_per_second=1.0,
                    items_per_second=1.0,
                    ms_per_batch=1.0 * 1e3,
                    ms_per_item=1.0 * 1e3,
                ),
            ),
            (
                dict(
                    batch_time=2.0,
                    batch_size=1,
                ),
                dict(
                    batch_time=2.0,
                    batch_size=1,
                    batches_per_second=0.5,
                    items_per_second=0.5,
                    ms_per_batch=2.0 * 1e3,
                    ms_per_item=2.0 * 1e3,
                ),
            ),
            (
                dict(
                    batch_time=1.0,
                    batch_size=2,
                ),
                dict(
                    batch_time=1.0,
                    batch_size=2,
                    batches_per_second=1.0,
                    items_per_second=2.0,
                    ms_per_batch=1.0 * 1e3,
                    ms_per_item=0.5 * 1e3,
                ),
            ),
        ],
    )
    def test_dict(self, inputs, expected_output):
        result = BatchBenchmarkResult(**inputs)
        assert result.dict() == expected_output


class TestBenchmarkResults:
    @pytest.mark.parametrize(
        "inputs",
        [
            (
                [
                    dict(
                        batch_time=random.random() + 1e-3,
                        batch_size=random.randint(1, 10),
                    )
                    for _ in range(1)
                ]
            ),
            (
                [
                    dict(
                        batch_time=random.random() + 1e-3,
                        batch_size=random.randint(1, 10),
                    )
                    for _ in range(0)
                ]
            ),
            (
                [
                    dict(
                        batch_time=random.random() + 1e-3,
                        batch_size=random.randint(1, 10),
                        inputs=[numpy.random.rand(10, 10)],
                        outputs=[numpy.random.rand(10, 10)],
                    )
                    for _ in range(10)
                ]
            ),
        ],
    )
    def test_append_and_retrievals(self, inputs):
        # Tests for append working correctly and properties
        # that retrieve from children work as intended
        results = BenchmarkResults()
        raw_results = []

        # Test appending and retrieving
        for index, result in enumerate(inputs):
            result = BatchBenchmarkResult(**result)
            results.append(result)
            raw_results.append(result)
            assert results[index] == result

        # Test retrieving the results
        assert results.results == raw_results

        # Test length of results
        assert len(results) == len(inputs)

        # Test number of batches added
        assert results.num_batches == len(inputs)

        # Test number of items added
        assert results.num_items == sum([result.batch_size for result in raw_results])

        # Test retreiving the batch times
        assert results.batch_times == [result.batch_time for result in raw_results]

        # Test retrieving the batch sizes
        assert results.batch_sizes == [result.batch_size for result in raw_results]

        # Test retrieving the inputs
        assert results.inputs == [result.inputs for result in raw_results]

        # Test retrieving the outputs
        assert results.outputs == [result.outputs for result in raw_results]

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                2,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                ],
                2.5,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=4,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=5,
                        batch_size=1,
                    ),
                ],
                3.5,
            ),
        ],
    )
    def test_batch_times_mean(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.batch_times_mean == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                2,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                ],
                2.5,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=4,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=5,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=10,
                        batch_size=1,
                    ),
                ],
                4,
            ),
        ],
    )
    def test_batch_times_median(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.batch_times_median == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=1,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                ],
                0.81650,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=4,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=5,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=10,
                        batch_size=1,
                    ),
                ],
                2.7857,
            ),
        ],
    )
    def test_batch_times_std(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert numpy.isclose(results.batch_times_std, expected_output)

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                0.5,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=1,
                    ),
                ],
                0.4,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=2,
                    ),
                    dict(
                        batch_time=3,
                        batch_size=3,
                    ),
                    dict(
                        batch_time=1,
                        batch_size=3,
                    ),
                ],
                0.5,
            ),
        ],
    )
    def test_batches_per_second(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.batches_per_second == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                0.5,
            ),
            (
                [
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                ],
                0.25,
            ),
            (
                [
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=32,
                        batch_size=16,
                    ),
                ],
                0.3,
            ),
        ],
    )
    def test_items_per_second(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.items_per_second == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                2 * 1e3,
            ),
            (
                [
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                ],
                64 * 1e3,
            ),
            (
                [
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=32,
                        batch_size=16,
                    ),
                ],
                160 / 3 * 1e3,
            ),
        ],
    )
    def test_ms_per_batch(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.ms_per_batch == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [],
                0,
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                2 * 1e3,
            ),
            (
                [
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                ],
                4 * 1e3,
            ),
            (
                [
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=64,
                        batch_size=16,
                    ),
                    dict(
                        batch_time=32,
                        batch_size=16,
                    ),
                ],
                10 / 3 * 1e3,
            ),
        ],
    )
    def test_ms_per_item(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.ms_per_item == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            ([], 0),
            (
                [dict(batch_time=1, batch_size=16)] * 90
                + [dict(batch_time=1000, batch_size=16)] * 10,
                1.0,
            ),
            (
                [dict(batch_time=1, batch_size=16)] * 44
                + [dict(batch_time=20, batch_size=16)] * 2
                + [dict(batch_time=100, batch_size=16)] * 44
                + [dict(batch_time=1000, batch_size=16)] * 10,
                20,
            ),
        ],
    )
    def test_batch_times_median_90(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.batch_times_median_90 == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            ([], 0),
            (
                [dict(batch_time=1, batch_size=16)] * 95
                + [dict(batch_time=1000, batch_size=16)] * 5,
                1.0,
            ),
            (
                [dict(batch_time=1, batch_size=16)] * 47
                + [dict(batch_time=20, batch_size=16)] * 1
                + [dict(batch_time=100, batch_size=16)] * 47
                + [dict(batch_time=1000, batch_size=16)] * 5,
                20,
            ),
        ],
    )
    def test_batch_times_median_95(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.batch_times_median_95 == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            ([], 0),
            (
                [dict(batch_time=1, batch_size=16)] * 99
                + [dict(batch_time=1000, batch_size=16)] * 1,
                1.0,
            ),
            (
                [dict(batch_time=1, batch_size=16)] * 49
                + [dict(batch_time=20, batch_size=16)] * 1
                + [dict(batch_time=100, batch_size=16)] * 49
                + [dict(batch_time=1000, batch_size=16)] * 1,
                20,
            ),
        ],
    )
    def test_batch_times_median_99(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.batch_times_median_99 == expected_output

    @pytest.mark.parametrize(
        "inputs,expected_output",
        [
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=1,
                    )
                ],
                dict(
                    results=[
                        dict(
                            batch_time=2,
                            batch_size=1,
                            batches_per_second=0.5,
                            items_per_second=0.5,
                            ms_per_batch=2000,
                            ms_per_item=2000,
                        )
                    ],
                    batch_times_mean=2.0,
                    batch_times_median=2.0,
                    batch_times_std=0.0,
                    batch_times_median_90=2.0,
                    batch_times_median_95=2.0,
                    batch_times_median_99=2.0,
                    batches_per_second=0.5,
                    items_per_second=0.5,
                    ms_per_batch=2000,
                    ms_per_item=2000,
                    num_items=1,
                    num_batches=1,
                ),
            ),
            (
                [
                    dict(
                        batch_time=2,
                        batch_size=32,
                    ),
                    dict(
                        batch_time=4,
                        batch_size=32,
                    ),
                    dict(
                        batch_time=8,
                        batch_size=32,
                    ),
                ],
                dict(
                    results=[
                        dict(
                            batch_time=2.0,
                            batches_per_second=0.5,
                            items_per_second=16.0,
                            ms_per_batch=2000.0,
                            ms_per_item=62.5,
                            batch_size=32,
                        ),
                        dict(
                            batch_time=4.0,
                            batches_per_second=0.25,
                            items_per_second=8.0,
                            ms_per_batch=4000.0,
                            ms_per_item=125.0,
                            batch_size=32,
                        ),
                        dict(
                            batch_time=8.0,
                            batches_per_second=0.125,
                            items_per_second=4.0,
                            ms_per_batch=8000.0,
                            ms_per_item=250.0,
                            batch_size=32,
                        ),
                    ],
                    batch_times_mean=14 / 3,
                    batch_times_median=4.0,
                    batch_times_std=2.494438257849294,
                    batch_times_median_90=3.0,
                    batch_times_median_95=3.0,
                    batch_times_median_99=3.0,
                    batches_per_second=0.21428571428571427,
                    items_per_second=6.857142857142857,
                    ms_per_batch=14 / 3 * 1e3,
                    ms_per_item=14 * 1e3 / (96),
                    num_items=96,
                    num_batches=3,
                ),
            ),
        ],
    )
    def test_dict(self, inputs, expected_output):
        results = BenchmarkResults()

        for result in inputs:
            result = BatchBenchmarkResult(**result)
            results.append(result)

        assert results.dict() == expected_output
