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
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "core_count": 1,
                "batch_size": 3,
                "inference_engine": "neural_magic",
                "inference_model_optimization": "optimization",
                "measurements": [0.1, 0.2, 0.3],
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
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
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
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
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "benchmarks": [
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                        "batch_size": 3,
                        "measurements": [0.1, 0.2, 0.3],
                        "core_count": 1,
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
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
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
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": None,
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "result": {
                    "benchmarks": [
                        {
                            "core_count": 1,
                            "batch_size": 1,
                            "inference_engine": "neural_magic",
                            "inference_model_optimization": "optimization",
                            "measurements": [0.1, 0.2, 0.3],
                            "iterations_per_check": 30,
                            "warmup_iterations_per_check": 30,
                        }
                    ]
                },
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
                "batch_sizes": [3, 6, 9],
                "source": "generated",
                "instruction_sets": ["AVX2"],
                "core_counts": [1, 2, 3],
                "project_id": "project id",
                "inference_models": [
                    {
                        "inference_model_optimization": None,
                        "inference_engine": "ort_gpu",
                    },
                    {
                        "inference_model_optimization": None,
                        "inference_engine": "neural_magic",
                    },
                ],
                "result": {
                    "benchmarks": [
                        {
                            "inference_model_optimization": "optimization",
                            "core_count": 1,
                            "measurements": [0.1, 0.2, 0.3],
                            "inference_engine": "neural_magic",
                            "batch_size": 1,
                        }
                    ]
                },
                "job": None,
                "name": "name",
                "benchmark_id": "benchmark id",
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_models": [
                    {
                        "inference_engine": "fail",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["fail"],
                "source": "fail",
                "job": None,
                "result": {
                    "benchmarks": [
                        {
                            "core_count": 1,
                            "batch_size": 1,
                            "inference_engine": "neural_magic",
                            "inference_model_optimization": "optimization",
                            "measurements": [0.1, 0.2, 0.3],
                            "iterations_per_check": 30,
                            "warmup_iterations_per_check": 30,
                        }
                    ]
                },
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            None,
            ["inference_models", "instruction_sets", "source"],
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
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "benchmark": {
                    "benchmark_id": "benchmark id",
                    "project_id": "project id",
                    "name": None,
                    "inference_models": [
                        {
                            "inference_engine": "ort_gpu",
                            "inference_model_optimization": None,
                        },
                        {
                            "inference_engine": "neural_magic",
                            "inference_model_optimization": None,
                        },
                    ],
                    "core_counts": [1, 2, 3],
                    "batch_sizes": [3, 6, 9],
                    "instruction_sets": ["AVX2"],
                    "source": None,
                    "job": None,
                    "result": None,
                    "iterations_per_check": 30,
                    "warmup_iterations_per_check": 30,
                }
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
                "result": {
                    "benchmarks": [
                        {
                            "core_count": 1,
                            "batch_size": 1,
                            "inference_engine": "neural_magic",
                            "inference_model_optimization": "optimization",
                            "measurements": [0.1, 0.2, 0.3],
                            "iterations_per_check": 30,
                            "warmup_iterations_per_check": 30,
                        }
                    ]
                },
            },
            {
                "benchmark": {
                    "source": "generated",
                    "job": None,
                    "name": "name",
                    "project_id": "project id",
                    "instruction_sets": ["AVX2"],
                    "iterations_per_check": 30,
                    "core_counts": [1, 2, 3],
                    "inference_models": [
                        {
                            "inference_model_optimization": None,
                            "inference_engine": "ort_gpu",
                        },
                        {
                            "inference_model_optimization": None,
                            "inference_engine": "neural_magic",
                        },
                    ],
                    "batch_sizes": [3, 6, 9],
                    "warmup_iterations_per_check": 30,
                    "result": {
                        "benchmarks": [
                            {
                                "inference_engine": "neural_magic",
                                "measurements": [0.1, 0.2, 0.3],
                                "batch_size": 1,
                                "core_count": 1,
                                "inference_model_optimization": "optimization",
                            }
                        ]
                    },
                    "benchmark_id": "benchmark id",
                    "created": "2020-10-29T15:39:07.437680",
                }
            },
            None,
        ),
        (
            {
                "benchmark_id": "benchmark id",
                "project_id": "project id",
                "name": "name",
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["fail"],
                "source": "fail",
                "job": None,
                "result": {
                    "benchmarks": [
                        {
                            "core_count": 1,
                            "batch_size": 1,
                            "inference_engine": "neural_magic",
                            "inference_model_optimization": "optimization",
                            "measurements": [0.1, 0.2, 0.3],
                            "iterations_per_check": 30,
                            "warmup_iterations_per_check": 30,
                        }
                    ]
                },
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
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": None,
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": None,
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": None,
                "job": None,
                "result": None,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "benchmarks": [
                    {
                        "benchmark_id": "benchmark id",
                        "project_id": "project id",
                        "name": None,
                        "inference_models": [
                            {
                                "inference_engine": "ort_gpu",
                                "inference_model_optimization": None,
                            },
                            {
                                "inference_engine": "neural_magic",
                                "inference_model_optimization": None,
                            },
                        ],
                        "core_counts": [1, 2, 3],
                        "batch_sizes": [3, 6, 9],
                        "instruction_sets": ["AVX2"],
                        "source": None,
                        "job": None,
                        "result": None,
                        "iterations_per_check": 30,
                        "warmup_iterations_per_check": 30,
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
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": "optimization",
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["AVX2"],
                "source": "generated",
                "job": None,
                "result": {
                    "benchmarks": [
                        {
                            "core_count": 1,
                            "batch_size": 1,
                            "inference_engine": "neural_magic",
                            "inference_model_optimization": "optimization",
                            "measurements": [0.1, 0.2, 0.3],
                            "iterations_per_check": 30,
                            "warmup_iterations_per_check": 30,
                        }
                    ]
                },
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "benchmarks": [
                    {
                        "benchmark_id": "benchmark id",
                        "project_id": "project id",
                        "name": "name",
                        "inference_models": [
                            {
                                "inference_engine": "ort_gpu",
                                "inference_model_optimization": "optimization",
                            },
                            {
                                "inference_engine": "neural_magic",
                                "inference_model_optimization": "optimization",
                            },
                        ],
                        "core_counts": [1, 2, 3],
                        "batch_sizes": [3, 6, 9],
                        "instruction_sets": ["AVX2"],
                        "source": "generated",
                        "job": None,
                        "result": {
                            "benchmarks": [
                                {
                                    "core_count": 1,
                                    "batch_size": 1,
                                    "inference_engine": "neural_magic",
                                    "inference_model_optimization": "optimization",
                                    "measurements": [0.1, 0.2, 0.3],
                                }
                            ]
                        },
                        "iterations_per_check": 30,
                        "warmup_iterations_per_check": 30,
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
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": "optimization",
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "instruction_sets": ["fail"],
                "source": "fail",
                "job": None,
                "result": "success",
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
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
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": "optimization",
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "name": None,
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": "optimization",
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            None,
        ),
        (
            {
                "name": "benchmark",
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": "optimization",
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            {
                "name": "benchmark",
                "inference_models": [
                    {
                        "inference_engine": "ort_gpu",
                        "inference_model_optimization": "optimization",
                    },
                    {
                        "inference_engine": "neural_magic",
                        "inference_model_optimization": "optimization",
                    },
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            None,
        ),
        (
            {
                "name": "benchmark",
                "inference_models": [
                    {
                        "inference_engine": "fail",
                        "inference_model_optimization": "optimization",
                    }
                ],
                "core_counts": [1, 2, 3],
                "batch_sizes": [3, 6, 9],
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 30,
            },
            None,
            ["inference_models"],
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
        ResponseProjectBenchmarkDeletedSchema,
        expected_input,
        expected_output,
    )
