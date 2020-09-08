from typing import Dict, List, Union
import pytest
from datetime import datetime

from neuralmagicML.server.schemas.projects_benchmarks import (
    ProjectBenchmarkResultSchema,
    ProjectBenchmarkResultsSchema,
    ProjectBenchmarkSchema,
    CreateProjectBenchmarkSchema,
    ResponseProjectBenchmarkSchema,
    ResponseProjectBenchmarksSchema,
    ResponseProjectBenchmarkDeletedSchema,
)

from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "core_count": 1,
                "batch_size": 3,
                "inference_engine": "neural_magic",
                "inference_model_optimization": "optimization",
                "measurements": [0.1, 0.2, 0.3],
            },
            {
                "core_count": 1,
                "batch_size": 3,
                "inference_engine": "neural_magic",
                "inference_model_optimization": "optimization",
                "measurements": [0.1, 0.2, 0.3],
            },
            None,
        ),
        (
            {
                "core_count": 1,
                "batch_size": 3,
                "inference_engine": "fail",
                "inference_model_optimization": "optimization",
                "measurements": [0.1, 0.2, 0.3],
            },
            None,
            ["inference_engine"],
        ),
    ],
)
def test_project_benchmark_result_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ProjectBenchmarkResultSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "core_count": 1,
                "batch_size": 3,
                "inference_engine": "neural_magic",
                "inference_model_optimization": "optimization",
                "measurements": [0.1, 0.2, 0.3],
            },
            {
                "benchmarks": [
                    {
                        "core_count": 1,
                        "batch_size": 3,
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                        "measurements": [0.1, 0.2, 0.3],
                    }
                ]
            },
            None,
        ),
        (
            {
                "core_count": 1,
                "batch_size": 3,
                "inference_engine": "fail",
                "inference_model_optimization": "optimization",
                "measurements": [0.1, 0.2, 0.3],
            },
            None,
            ["benchmarks"],
        ),
    ],
)
def test_project_benchmark_result_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ProjectBenchmarkResultsSchema,
        {"benchmarks": [expected_input]},
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": None,
                "inference_engine": "neural_magic",
                "inference_model_optimization": None,
                "comparison_engine": None,
                "comparison_model_optimization": None,
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
            },
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": None,
                "inference_engine": "neural_magic",
                "inference_model_optimization": None,
                "comparison_engine": None,
                "comparison_model_optimization": None,
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "ort_cpu",
                "inference_model_optimization": "optimization",
                "comparison_engine": "ort_cpu",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "result": "success",
            },
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "ort_cpu",
                "inference_model_optimization": "optimization",
                "comparison_engine": "ort_cpu",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "result": "success",
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "fail",
                "inference_model_optimization": "optimization",
                "comparison_engine": "fail",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["fail"],
                "source": "fail",
                "job": None,
                "result": "success",
            },
            None,
            ["inference_engine", "comparison_engine", "instruction_sets", "source"],
        ),
    ],
)
def test_project_benchmark_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["created"] = created.isoformat()
    schema_tester(
        ProjectBenchmarkSchema, expected_input, expected_output, expect_validation_error
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": None,
                "inference_engine": "neural_magic",
                "inference_model_optimization": None,
                "comparison_engine": None,
                "comparison_model_optimization": None,
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
            },
            {
                "benchmark": {
                    "benchmark_id": "benchmark id",
                    "project_id": "project id",
                    "name": None,
                    "inference_engine": "neural_magic",
                    "inference_model_optimization": None,
                    "comparison_engine": None,
                    "comparison_model_optimization": None,
                    "core_counts": [1, 2, 3],
                    "batch_sizes": [3, 6, 9],
                    "instruction_sets": ["AVX2"],
                    "source": None,
                    "job": None,
                    "result": None,
                }
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "ort_cpu",
                "inference_model_optimization": "optimization",
                "comparison_engine": "ort_cpu",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "result": "success",
            },
            {
                "benchmark": {
                    "benchmark_id": "benchmark id",
                    "project_id": "project id",
                    "name": "name",
                    "inference_engine": "ort_cpu",
                    "inference_model_optimization": "optimization",
                    "comparison_engine": "ort_cpu",
                    "comparison_model_optimization": "comparison",
                    "core_counts": [1, 2, 3],
                    "batch_sizes": [3, 6, 9],
                    "instruction_sets": ["AVX2"],
                    "source": "generated",
                    "job": None,
                    "result": "success",
                }
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "fail",
                "inference_model_optimization": "optimization",
                "comparison_engine": "fail",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["fail"],
                "source": "fail",
                "job": None,
                "result": "success",
            },
            None,
            ["benchmark"],
        ),
    ],
)
def test_response_project_benchmark_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["benchmark"]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectBenchmarkSchema,
        {"benchmark": expected_input},
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": None,
                "inference_engine": "neural_magic",
                "inference_model_optimization": None,
                "comparison_engine": None,
                "comparison_model_optimization": None,
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
            },
            {
                "benchmarks": [
                    {
                        "benchmark_id": "benchmark id",
                        "project_id": "project id",
                        "name": None,
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                        "comparison_engine": None,
                        "comparison_model_optimization": None,
                        "core_counts": [1, 2, 3],
                        "batch_sizes": [3, 6, 9],
                        "instruction_sets": ["AVX2"],
                        "source": None,
                        "job": None,
                        "result": None,
                    }
                ]
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "ort_cpu",
                "inference_model_optimization": "optimization",
                "comparison_engine": "ort_cpu",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "result": "success",
            },
            {
                "benchmarks": [
                    {
                        "benchmark_id": "benchmark id",
                        "project_id": "project id",
                        "name": "name",
                        "inference_engine": "ort_cpu",
                        "inference_model_optimization": "optimization",
                        "comparison_engine": "ort_cpu",
                        "comparison_model_optimization": "comparison",
                        "core_counts": [1, 2, 3],
                        "batch_sizes": [3, 6, 9],
                        "instruction_sets": ["AVX2"],
                        "source": "generated",
                        "job": None,
                        "result": "success",
                    }
                ]
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_engine": "fail",
                "inference_model_optimization": "optimization",
                "comparison_engine": "fail",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["fail"],
                "source": "fail",
                "job": None,
                "result": "success",
            },
            None,
            ["benchmarks"],
        ),
    ],
)
def test_response_project_benchmarks_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["benchmarks"][0]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectBenchmarksSchema,
        {"benchmarks": [expected_input]},
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "inference_engine": "ort_gpu",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
            },
            {
                "name": None,
                "inference_engine": "ort_gpu",
                "inference_model_optimization": None,
                "comparison_engine": None,
                "comparison_model_optimization": None,
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
            },
            None,
        ),
        (
            {
                "name": "benchmark",
                "inference_engine": "ort_gpu",
                "inference_model_optimization": "optimization",
                "comparison_engine": "ort_cpu",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
            },
            {
                "name": "benchmark",
                "inference_engine": "ort_gpu",
                "inference_model_optimization": "optimization",
                "comparison_engine": "ort_cpu",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
            },
            None,
        ),
        (
            {
                "name": "benchmark",
                "inference_engine": "fail",
                "inference_model_optimization": "optimization",
                "comparison_engine": "fail",
                "comparison_model_optimization": "comparison",
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
            },
            None,
            ["inference_engine", "comparison_engine"],
        ),
    ],
)
def test_create_project_benchmark_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        CreateProjectBenchmarkSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"project_id": "project id", "benchmark_id": "benchmark id"},
            {
                "success": True,
                "project_id": "project id",
                "benchmark_id": "benchmark id",
            },
        ),
        (
            {
                "success": False,
                "project_id": "project id",
                "benchmark_id": "benchmark id",
            },
            {
                "success": False,
                "project_id": "project id",
                "benchmark_id": "benchmark id",
            },
        ),
    ],
)
def test_response_project_benchmark_deleted_schema(
    expected_input: Dict, expected_output: Dict
):
    schema_tester(
        ResponseProjectBenchmarkDeletedSchema, expected_input, expected_output,
    )
