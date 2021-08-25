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

from typing import Any, Dict

import pytest

from sparseml.benchmark.serialization import (
    BatchBenchmarkResult,
    BenchmarkConfig,
    BenchmarkInfo,
    BenchmarkResult,
)


def _test_serialization(
    SchemaType: type, data: Dict[str, Any], expect_error: bool = False
):
    exception = None
    try:
        # test construction
        schema = SchemaType(**data)

        if expect_error:
            assert False, "Expected error for schema construction"

        schema.json()

        # test serialization
        schema_str = schema.json()
        assert schema_str, "No json returned for serialization"
    except Exception as e:
        if not expect_error:
            assert False, "Unexpected error: {}".format(e)
        exception = e

    if expect_error:
        assert exception is not None, "Expected error for schema construction"


# Test serialization of BatchBenchmarkResult
@pytest.mark.parametrize(
    "data,expect_error",
    [
        (
            dict(
                batch_time=0.5,
                batches_per_second=2,
                items_per_second=20,
                ms_per_batch=20,
                ms_per_item=20,
                batch_size=10,
            ),
            False,
        ),
        (
            dict(
                batch_time=1e-10,
                batches_per_second=1e10,
                items_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                batch_size=1,
            ),
            False,
        ),
        (
            dict(
                batch_time=0,
                batches_per_second=2,
                items_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                batch_size=1,
            ),
            True,
        ),
        (
            dict(
                batch_time=1e-10,
                batches_per_second=2,
                items_per_second=-1,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                batch_size=1,
            ),
            True,
        ),
        (
            dict(
                batch_time=1e-10,
                batches_per_second=2,
                items_per_second=1e-10,
                ms_per_batch=-1,
                ms_per_item=1e-10,
                batch_size=1,
            ),
            True,
        ),
        (
            dict(
                batch_time=1e-10,
                batches_per_second=2,
                items_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=-1,
                batch_size=1,
            ),
            True,
        ),
        (
            dict(
                batch_time=1e-10,
                batches_per_second=2,
                items_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                batch_size=0,
            ),
            True,
        ),
    ],
)
def test_batch_benchmark_result_schema(data: Dict[str, Any], expect_error: bool):
    _test_serialization(BatchBenchmarkResult, data, expect_error)


@pytest.mark.parametrize(
    "results,expected_output",
    [
        (
            dict(
                batch_time=2,
                batch_size=1,
            ),
            dict(
                batch_time=2,
                batch_size=1,
                batches_per_second=0.5,
                items_per_second=0.5,
                ms_per_batch=2000,
                ms_per_item=2000,
            ),
        ),
        (
            dict(
                batch_time=2,
                batch_size=32,
            ),
            dict(
                batch_time=2.0,
                batches_per_second=0.5,
                items_per_second=16.0,
                ms_per_batch=2000.0,
                ms_per_item=62.5,
                batch_size=32,
            ),
        ),
        (
            dict(
                batch_time=4,
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
        ),
        (
            dict(
                batch_time=8,
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
        ),
    ],
)
def test_batch_benchmark_result_from_result(
    results: Dict[str, Any], expected_output: Dict[str, Any]
):
    benchmark_result = BatchBenchmarkResult.from_result(**results)
    assert benchmark_result.dict() == expected_output


# Test serialization of BenchmarkResult
@pytest.mark.parametrize(
    "data,expect_error",
    [
        (
            dict(
                batch_times_mean=0.5,
                batch_times_median=0.5,
                batch_times_std=0.5,
                batch_times_median_90=0.5,
                batch_times_median_95=0.5,
                batch_times_median_99=0.5,
                items_per_second=20,
                batches_per_second=20,
                ms_per_batch=20,
                ms_per_item=20,
                num_items=10,
                num_batches=10,
                results=[],
            ),
            False,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            False,
        ),
        (
            dict(
                batch_times_mean=0,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=0,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=0,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=0,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=0,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=0,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=0,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=0,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=0,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=0,
                num_items=1,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=0,
                num_batches=1,
                results=[],
            ),
            True,
        ),
        (
            dict(
                batch_times_mean=1e-10,
                batch_times_median=1e-10,
                batch_times_std=1e-10,
                batch_times_median_90=1e-10,
                batch_times_median_95=1e-10,
                batch_times_median_99=1e-10,
                items_per_second=1e-10,
                batches_per_second=1e-10,
                ms_per_batch=1e-10,
                ms_per_item=1e-10,
                num_items=1,
                num_batches=0,
                results=[],
            ),
            True,
        ),
    ],
)
def test_benchmark_result_schema(data: Dict[str, Any], expect_error: bool):
    _test_serialization(BenchmarkResult, data, expect_error)


@pytest.mark.parametrize(
    "results,expected_output",
    [
        (
            dict(
                results=[
                    BatchBenchmarkResult.from_result(
                        batch_time=2,
                        batch_size=1,
                    )
                ]
            ),
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
            dict(
                results=[
                    BatchBenchmarkResult.from_result(
                        batch_time=2,
                        batch_size=32,
                    ),
                    BatchBenchmarkResult.from_result(
                        batch_time=4,
                        batch_size=32,
                    ),
                    BatchBenchmarkResult.from_result(
                        batch_time=8,
                        batch_size=32,
                    ),
                ]
            ),
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
def test_benchmark_result_from_results(
    results: Dict[str, Any], expected_output: Dict[str, Any]
):
    schema = BenchmarkResult.from_results(**results)
    assert schema.dict() == expected_output


# Test serialization of BenchmarkConfig
@pytest.mark.parametrize(
    "data,expect_error",
    [
        (
            dict(
                batch_size=32,
                iterations=30,
                warmup_iterations=10,
                device="cpu",
                framework_args=dict(
                    arg="val",
                ),
                inference_provider=dict(
                    name="cpu",
                    description="Base CPU provider within ONNXRuntime",
                    device="cpu",
                    supported_sparsification={"modifiers": []},
                    available=True,
                    properties={},
                    warnings=[],
                ),
            ),
            False,
        ),
        (
            dict(
                batch_size=1,
                iterations=0,
                warmup_iterations=0,
                device="cpu",
                framework_args={},
                inference_provider=dict(
                    name="cpu",
                    description="Base CPU provider within ONNXRuntime",
                    device="cpu",
                    supported_sparsification={"modifiers": []},
                    available=True,
                    properties={},
                    warnings=[],
                ),
            ),
            False,
        ),
        (
            dict(
                batch_size=1,
                iterations=0,
                warmup_iterations=0,
                device="cpu",
                framework_args={},
                inference_provider={},
            ),
            True,
        ),
        (
            dict(
                batch_size=0,
                iterations=0,
                warmup_iterations=0,
                device="cpu",
                framework_args={},
                inference_provider=dict(
                    name="cpu",
                    description="Base CPU provider within ONNXRuntime",
                    device="cpu",
                    supported_sparsification={"modifiers": []},
                    available=True,
                    properties={},
                    warnings=[],
                ),
            ),
            True,
        ),
        (
            dict(
                batch_size=1,
                iterations=-1,
                warmup_iterations=0,
                device="cpu",
                framework_args={},
                inference_provider=dict(
                    name="cpu",
                    description="Base CPU provider within ONNXRuntime",
                    device="cpu",
                    supported_sparsification={"modifiers": []},
                    available=True,
                    properties={},
                    warnings=[],
                ),
            ),
            True,
        ),
        (
            dict(
                batch_size=1,
                iterations=0,
                warmup_iterations=-1,
                device="cpu",
                framework_args={},
                inference_provider=dict(
                    name="cpu",
                    description="Base CPU provider within ONNXRuntime",
                    device="cpu",
                    supported_sparsification={"modifiers": []},
                    available=True,
                    properties={},
                    warnings=[],
                ),
            ),
            True,
        ),
    ],
)
def test_benchmark_config(data: Dict[str, Any], expect_error: bool):
    _test_serialization(BenchmarkConfig, data, expect_error)


# Test serialization of BenchmarkInfo
@pytest.mark.parametrize(
    "data,expect_error",
    [
        (
            dict(
                framework="onnx",
                package_versions=dict(
                    onnx="1.7.0",
                    onnxruntime="1.8.0",
                    sparsezoo=None,
                    sparseml="0.2.0",
                ),
                benchmark=dict(
                    results=[
                        dict(
                            batch_time=0.004634857177734375,
                            batches_per_second=215.75637860082304,
                            items_per_second=215.75637860082304,
                            ms_per_batch=4.634857177734375,
                            ms_per_item=4.634857177734375,
                            batch_size=1,
                        ),
                        dict(
                            batch_time=0.0023450851440429688,
                            batches_per_second=426.4237494916633,
                            items_per_second=426.4237494916633,
                            ms_per_batch=2.3450851440429688,
                            ms_per_item=2.3450851440429688,
                            batch_size=1,
                        ),
                        dict(
                            batch_time=0.002274751663208008,
                            batches_per_second=439.6084267896447,
                            items_per_second=439.6084267896447,
                            ms_per_batch=2.274751663208008,
                            ms_per_item=2.274751663208008,
                            batch_size=1,
                        ),
                        dict(
                            batch_time=0.002254486083984375,
                            batches_per_second=443.5600676818951,
                            items_per_second=443.5600676818951,
                            ms_per_batch=2.254486083984375,
                            ms_per_item=2.254486083984375,
                            batch_size=1,
                        ),
                        dict(
                            batch_time=0.0022568702697753906,
                            batches_per_second=443.0914853158673,
                            items_per_second=443.0914853158673,
                            ms_per_batch=2.2568702697753906,
                            ms_per_item=2.2568702697753906,
                            batch_size=1,
                        ),
                    ],
                    batch_times_mean=0.0027532100677490233,
                    batch_times_median=0.002274751663208008,
                    batch_times_std=0.0009413992831712611,
                    batch_times_median_90=0.0022658109664916992,
                    batch_times_median_95=0.0022658109664916992,
                    batch_times_median_99=0.0022658109664916992,
                    items_per_second=363.2123867749701,
                    batches_per_second=363.2123867749701,
                    ms_per_batch=2.7532100677490234,
                    ms_per_item=2.7532100677490234,
                    num_items=5,
                    num_batches=5,
                ),
                config=dict(
                    batch_size=1,
                    iterations=5,
                    warmup_iterations=0,
                    device="cpu",
                    framework_args={},
                    inference_provider=dict(
                        name="cpu",
                        description="Base CPU provider within ONNXRuntime",
                        device="cpu",
                        supported_sparsification={"modifiers": []},
                        available=True,
                        properties={},
                        warnings=[],
                    ),
                ),
            ),
            False,
        ),
    ],
)
def test_benchmark_info(data: Dict[str, Any], expect_error: bool):
    _test_serialization(BenchmarkInfo, data, expect_error)
