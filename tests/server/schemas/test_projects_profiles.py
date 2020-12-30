from typing import Dict, List, Union
import pytest
from datetime import datetime

from sparseml.server.schemas.projects_profiles import (
    ProjectProfileMeasurementSchema,
    ProjectProfileMeasurementsSchema,
    ProjectProfileOpSchema,
    ProjectProfileOpMeasurementsSchema,
    ProjectProfileOpBaselineMeasurementSchema,
    ProjectProfileModelOpsMeasurementsSchema,
    ProjectProfileModelOpsBaselineMeasurementsSchema,
    ProjectProfileAnalysisSchema,
    ProjectProfileSchema,
    ProjectLossProfileSchema,
    ProjectPerfProfileSchema,
    CreateProjectPerfProfileSchema,
    CreateProjectLossProfileSchema,
    SearchProjectProfilesSchema,
    ResponseProjectLossProfileSchema,
    ResponseProjectLossProfilesSchema,
    ResponseProjectPerfProfileSchema,
    ResponseProjectPerfProfilesSchema,
    ResponseProjectProfileDeletedSchema,
)
from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [({"measurement": 1.0}, {"measurement": 1.0})],
)
def test_project_profile_measurment_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileMeasurementSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"baseline_measurement_key": None, "measurements": {None: 0}},
            {"baseline_measurement_key": None, "measurements": {None: 0}},
        ),
        (
            {"baseline_measurement_key": "0.0", "measurements": {"0.0": 1.0}},
            {"baseline_measurement_key": "0.0", "measurements": {"0.0": 1.0}},
        ),
    ],
)
def test_project_profile_measurments_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileMeasurementsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"id": None, "name": None, "index": None},
            {"id": None, "name": None, "index": None},
        ),
        (
            {"id": "id", "name": "conv", "index": 0},
            {"id": "id", "name": "conv", "index": 0},
        ),
    ],
)
def test_project_profile_op_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileOpSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "id": None,
                "name": None,
                "index": None,
                "baseline_measurement_key": None,
                "measurements": {None: 0},
            },
            {
                "id": None,
                "name": None,
                "index": None,
                "baseline_measurement_key": None,
                "measurements": {None: 0},
            },
        ),
        (
            {
                "id": "id",
                "name": "conv",
                "index": 0,
                "baseline_measurement_key": "0.0",
                "measurements": {"0.0": 1.0},
            },
            {
                "id": "id",
                "name": "conv",
                "index": 0,
                "baseline_measurement_key": "0.0",
                "measurements": {"0.0": 1.0},
            },
        ),
    ],
)
def test_project_profile_op_measurements_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileOpMeasurementsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"id": None, "name": None, "index": None, "measurement": 1.0},
            {"id": None, "name": None, "index": None, "measurement": 1.0},
        ),
        (
            {"id": "id", "name": "conv", "index": 0, "measurement": 1.0},
            {"id": "id", "name": "conv", "index": 0, "measurement": 1.0},
        ),
    ],
)
def test_project_profile_op_baseline_measurement_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileOpBaselineMeasurementSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({"model": None, "ops": None}, {"model": None, "ops": None}),
        (
            {
                "model": {
                    "baseline_measurement_key": "0.0",
                    "measurements": {"0.0": 1.0},
                },
                "ops": [
                    {
                        "id": None,
                        "name": None,
                        "index": None,
                        "baseline_measurement_key": None,
                        "measurements": {None: 0},
                    },
                ],
            },
            {
                "model": {
                    "baseline_measurement_key": "0.0",
                    "measurements": {"0.0": 1.0},
                },
                "ops": [
                    {
                        "id": None,
                        "name": None,
                        "index": None,
                        "baseline_measurement_key": None,
                        "measurements": {None: 0},
                    },
                ],
            },
        ),
    ],
)
def test_project_profile_model_ops_measurments_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileModelOpsMeasurementsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({"model": None, "ops": None}, {"model": None, "ops": None}),
        (
            {
                "model": {
                    "measurement": 1.0,
                },
                "ops": [
                    {"id": None, "name": None, "index": None, "measurement": 1.0},
                ],
            },
            {
                "model": {"measurement": 1.0},
                "ops": [
                    {"id": None, "name": None, "index": None, "measurement": 1.0},
                ],
            },
        ),
    ],
)
def test_project_profile_model_ops_baseline_measurments_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileModelOpsBaselineMeasurementsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"baseline": None, "pruning": None, "quantization": None},
            {"baseline": None, "pruning": None, "quantization": None},
        ),
        (
            {
                "baseline": {
                    "model": {
                        "measurement": 1.0,
                    },
                    "ops": [
                        {"id": None, "name": None, "index": None, "measurement": 1.0},
                    ],
                },
                "pruning": {
                    "model": {
                        "baseline_measurement_key": "0.0",
                        "measurements": {"0.0": 1.0},
                    },
                    "ops": [
                        {
                            "id": "id",
                            "name": "conv",
                            "index": 0,
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                    ],
                },
                "quantization": {
                    "model": {
                        "baseline_measurement_key": "0.0",
                        "measurements": {"0.0": 1.0},
                    },
                    "ops": [
                        {
                            "id": "id",
                            "name": "conv",
                            "index": 0,
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                    ],
                },
            },
            {
                "baseline": {
                    "model": {"measurement": 1.0},
                    "ops": [
                        {"id": None, "name": None, "index": None, "measurement": 1.0},
                    ],
                },
                "pruning": {
                    "model": {
                        "baseline_measurement_key": "0.0",
                        "measurements": {"0.0": 1.0},
                    },
                    "ops": [
                        {
                            "id": "id",
                            "name": "conv",
                            "index": 0,
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                    ],
                },
                "quantization": {
                    "model": {
                        "baseline_measurement_key": "0.0",
                        "measurements": {"0.0": 1.0},
                    },
                    "ops": [
                        {
                            "id": "id",
                            "name": "conv",
                            "index": 0,
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                    ],
                },
            },
        ),
    ],
)
def test_project_profile_anaylsis_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectProfileAnalysisSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "name": None,
                "analysis": None,
            },
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "name": None,
                "analysis": None,
            },
            None,
        ),
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "uploaded",
                "job": None,
                "name": "profile",
                "analysis": {
                    "baseline": {
                        "model": {
                            "measurement": 1.0,
                        },
                        "ops": [
                            {
                                "id": None,
                                "name": None,
                                "index": None,
                                "measurement": 1.0,
                            },
                        ],
                    },
                    "pruning": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                    "quantization": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                },
            },
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "uploaded",
                "job": None,
                "name": "profile",
                "analysis": {
                    "baseline": {
                        "model": {"measurement": 1.0},
                        "ops": [
                            {
                                "id": None,
                                "name": None,
                                "index": None,
                                "measurement": 1.0,
                            },
                        ],
                    },
                    "pruning": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                    "quantization": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                },
            },
            None,
        ),
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "fail",
                "job": None,
                "name": "profile",
                "analysis": None,
            },
            None,
            ["source"],
        ),
    ],
)
def test_project_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["created"] = created.isoformat()
    schema_tester(
        ProjectProfileSchema, expected_input, expected_output, expect_validation_error
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "name": None,
                "analysis": None,
                "pruning_estimations": False,
                "pruning_estimation_type": "weight_magnitude",
                "pruning_structure": "channel",
                "quantized_estimations": False,
            },
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "name": None,
                "analysis": None,
                "pruning_estimations": False,
                "pruning_estimation_type": "weight_magnitude",
                "pruning_structure": "channel",
                "quantized_estimations": False,
            },
            None,
        ),
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "uploaded",
                "job": None,
                "name": "profile",
                "pruning_estimations": True,
                "pruning_estimation_type": "one_shot",
                "pruning_structure": "filter",
                "quantized_estimations": True,
                "analysis": {
                    "baseline": {
                        "model": {
                            "measurement": 1.0,
                        },
                        "ops": [
                            {
                                "id": None,
                                "name": None,
                                "index": None,
                                "measurement": 1.0,
                            },
                        ],
                    },
                    "pruning": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                    "quantization": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                },
            },
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "uploaded",
                "job": None,
                "name": "profile",
                "pruning_estimations": True,
                "pruning_estimation_type": "one_shot",
                "pruning_structure": "filter",
                "quantized_estimations": True,
                "analysis": {
                    "baseline": {
                        "model": {"measurement": 1.0},
                        "ops": [
                            {
                                "id": None,
                                "name": None,
                                "index": None,
                                "measurement": 1.0,
                            },
                        ],
                    },
                    "pruning": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                    "quantization": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                },
            },
            None,
        ),
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "fail",
                "job": None,
                "name": "profile",
                "analysis": None,
                "pruning_estimations": True,
                "pruning_estimation_type": "fail",
                "pruning_structure": "fail",
                "quantized_estimations": True,
            },
            None,
            ["source", "pruning_estimation_type", "pruning_structure"],
        ),
    ],
)
def test_project_loss_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["created"] = created.isoformat()
    schema_tester(
        ProjectLossProfileSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "name": None,
                "analysis": None,
                "batch_size": None,
                "core_count": None,
                "instruction_sets": None,
                "pruning_estimations": False,
                "quantized_estimations": False,
                "iterations_per_check": 0,
                "warmup_iterations_per_check": 0,
            },
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "name": None,
                "analysis": None,
                "batch_size": None,
                "core_count": None,
                "instruction_sets": None,
                "pruning_estimations": False,
                "quantized_estimations": False,
                "iterations_per_check": 0,
                "warmup_iterations_per_check": 0,
            },
            None,
        ),
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "uploaded",
                "job": None,
                "name": "profile",
                "batch_size": 64,
                "core_count": 4,
                "instruction_sets": ["AVX2", "AVX512", "VNNI"],
                "pruning_estimations": True,
                "quantized_estimations": True,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 5,
                "analysis": {
                    "baseline": {
                        "model": {
                            "measurement": 1.0,
                        },
                        "ops": [
                            {
                                "id": None,
                                "name": None,
                                "index": None,
                                "measurement": 1.0,
                            },
                        ],
                    },
                    "pruning": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                    "quantization": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                },
            },
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "uploaded",
                "job": None,
                "name": "profile",
                "batch_size": 64,
                "core_count": 4,
                "instruction_sets": ["AVX2", "AVX512", "VNNI"],
                "pruning_estimations": True,
                "quantized_estimations": True,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 5,
                "analysis": {
                    "baseline": {
                        "model": {"measurement": 1.0},
                        "ops": [
                            {
                                "id": None,
                                "name": None,
                                "index": None,
                                "measurement": 1.0,
                            },
                        ],
                    },
                    "pruning": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                    "quantization": {
                        "model": {
                            "baseline_measurement_key": "0.0",
                            "measurements": {"0.0": 1.0},
                        },
                        "ops": [
                            {
                                "id": "id",
                                "name": "conv",
                                "index": 0,
                                "baseline_measurement_key": "0.0",
                                "measurements": {"0.0": 1.0},
                            },
                        ],
                    },
                },
            },
            None,
        ),
        (
            {
                "profile_id": "profile id",
                "project_id": "project id",
                "source": "fail",
                "job": None,
                "name": "profile",
                "analysis": None,
                "batch_size": 64,
                "core_count": 4,
                "instruction_sets": "fail",
                "pruning_estimations": True,
                "quantized_estimations": True,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 5,
            },
            None,
            ["source", "instruction_sets"],
        ),
    ],
)
def test_project_perf_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["created"] = created.isoformat()
    schema_tester(
        ProjectPerfProfileSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {},
            {
                "name": None,
                "pruning_estimations": True,
                "pruning_estimation_type": "weight_magnitude",
                "pruning_structure": "unstructured",
                "quantized_estimations": False,
            },
            None,
        ),
        (
            {
                "name": "loss profile",
                "pruning_estimations": False,
                "pruning_estimation_type": "one_shot",
                "pruning_structure": "block_4",
                "quantized_estimations": True,
            },
            {
                "name": "loss profile",
                "pruning_estimations": False,
                "pruning_estimation_type": "one_shot",
                "pruning_structure": "block_4",
                "quantized_estimations": True,
            },
            None,
        ),
        (
            {
                "name": "loss profile",
                "pruning_estimations": False,
                "pruning_estimation_type": "fail",
                "pruning_structure": "fail",
                "quantized_estimations": True,
            },
            None,
            ["pruning_estimation_type", "pruning_structure"],
        ),
    ],
)
def test_create_project_loss_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        CreateProjectLossProfileSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {},
            {
                "name": None,
                "batch_size": 1,
                "core_count": -1,
                "pruning_estimations": True,
                "quantized_estimations": False,
                "iterations_per_check": 10,
                "warmup_iterations_per_check": 5,
            },
        ),
        (
            {
                "name": "perf profile",
                "batch_size": 32,
                "core_count": 4,
                "pruning_estimations": False,
                "quantized_estimations": True,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 25,
            },
            {
                "name": "perf profile",
                "batch_size": 32,
                "core_count": 4,
                "pruning_estimations": False,
                "quantized_estimations": True,
                "iterations_per_check": 30,
                "warmup_iterations_per_check": 25,
            },
        ),
    ],
)
def test_create_project_perf_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        CreateProjectPerfProfileSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {},
            {"page": 1, "page_length": 20},
            None,
        ),
        (
            {"page": 10, "page_length": 25},
            {"page": 10, "page_length": 25},
            None,
        ),
        (
            {"page": 0, "page_length": 0},
            None,
            ["page", "page_length"],
        ),
    ],
)
def test_create_project_loss_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        SearchProjectProfilesSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": None,
                    "job": None,
                    "name": None,
                    "analysis": None,
                    "pruning_estimations": False,
                    "pruning_estimation_type": "weight_magnitude",
                    "pruning_structure": "channel",
                    "quantized_estimations": False,
                }
            },
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": None,
                    "job": None,
                    "name": None,
                    "analysis": None,
                    "pruning_estimations": False,
                    "pruning_estimation_type": "weight_magnitude",
                    "pruning_structure": "channel",
                    "quantized_estimations": False,
                }
            },
            None,
        ),
        (
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": "uploaded",
                    "job": None,
                    "name": "profile",
                    "pruning_estimations": True,
                    "pruning_estimation_type": "one_shot",
                    "pruning_structure": "filter",
                    "quantized_estimations": True,
                    "analysis": None,
                }
            },
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": "uploaded",
                    "job": None,
                    "name": "profile",
                    "pruning_estimations": True,
                    "pruning_estimation_type": "one_shot",
                    "pruning_structure": "filter",
                    "quantized_estimations": True,
                    "analysis": None,
                }
            },
            None,
        ),
        (
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": "fail",
                    "job": None,
                    "name": "profile",
                    "analysis": None,
                    "pruning_estimations": True,
                    "pruning_estimation_type": "fail",
                    "pruning_structure": "fail",
                    "quantized_estimations": True,
                }
            },
            None,
            ["profile"],
        ),
    ],
)
def test_response_project_loss_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["profile"]["created"] = created
    if expected_output:
        expected_output["profile"]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectLossProfileSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": None,
                        "job": None,
                        "name": None,
                        "analysis": None,
                        "pruning_estimations": False,
                        "pruning_estimation_type": "weight_magnitude",
                        "pruning_structure": "channel",
                        "quantized_estimations": False,
                    }
                ]
            },
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": None,
                        "job": None,
                        "name": None,
                        "analysis": None,
                        "pruning_estimations": False,
                        "pruning_estimation_type": "weight_magnitude",
                        "pruning_structure": "channel",
                        "quantized_estimations": False,
                    }
                ]
            },
            None,
        ),
        (
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": "uploaded",
                        "job": None,
                        "name": "profile",
                        "pruning_estimations": True,
                        "pruning_estimation_type": "one_shot",
                        "pruning_structure": "filter",
                        "quantized_estimations": True,
                        "analysis": None,
                    }
                ]
            },
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": "uploaded",
                        "job": None,
                        "name": "profile",
                        "pruning_estimations": True,
                        "pruning_estimation_type": "one_shot",
                        "pruning_structure": "filter",
                        "quantized_estimations": True,
                        "analysis": None,
                    }
                ]
            },
            None,
        ),
        (
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": "fail",
                        "job": None,
                        "name": "profile",
                        "analysis": None,
                        "pruning_estimations": True,
                        "pruning_estimation_type": "fail",
                        "pruning_structure": "fail",
                        "quantized_estimations": True,
                    }
                ]
            },
            None,
            ["profiles"],
        ),
    ],
)
def test_response_project_loss_profiles_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["profiles"][0]["created"] = created
    if expected_output:
        expected_output["profiles"][0]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectLossProfilesSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": None,
                    "job": None,
                    "name": None,
                    "analysis": None,
                    "batch_size": None,
                    "core_count": None,
                    "instruction_sets": None,
                    "pruning_estimations": False,
                    "quantized_estimations": False,
                    "iterations_per_check": 0,
                    "warmup_iterations_per_check": 0,
                }
            },
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": None,
                    "job": None,
                    "name": None,
                    "analysis": None,
                    "batch_size": None,
                    "core_count": None,
                    "instruction_sets": None,
                    "pruning_estimations": False,
                    "quantized_estimations": False,
                    "iterations_per_check": 0,
                    "warmup_iterations_per_check": 0,
                }
            },
            None,
        ),
        (
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": "uploaded",
                    "job": None,
                    "name": "profile",
                    "batch_size": 64,
                    "core_count": 4,
                    "instruction_sets": ["AVX2", "AVX512", "VNNI"],
                    "pruning_estimations": True,
                    "quantized_estimations": True,
                    "iterations_per_check": 30,
                    "warmup_iterations_per_check": 5,
                    "analysis": None,
                }
            },
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": "uploaded",
                    "job": None,
                    "name": "profile",
                    "batch_size": 64,
                    "core_count": 4,
                    "instruction_sets": ["AVX2", "AVX512", "VNNI"],
                    "pruning_estimations": True,
                    "quantized_estimations": True,
                    "iterations_per_check": 30,
                    "warmup_iterations_per_check": 5,
                    "analysis": None,
                }
            },
            None,
        ),
        (
            {
                "profile": {
                    "profile_id": "profile id",
                    "project_id": "project id",
                    "source": "fail",
                    "job": None,
                    "name": "profile",
                    "analysis": None,
                    "batch_size": 64,
                    "core_count": 4,
                    "instruction_sets": "fail",
                    "pruning_estimations": True,
                    "quantized_estimations": True,
                    "iterations_per_check": 30,
                    "warmup_iterations_per_check": 5,
                }
            },
            None,
            ["profile"],
        ),
    ],
)
def test_repsonse_project_perf_profile_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["profile"]["created"] = created
    if expected_output:
        expected_output["profile"]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectPerfProfileSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": None,
                        "job": None,
                        "name": None,
                        "analysis": None,
                        "batch_size": None,
                        "core_count": None,
                        "instruction_sets": None,
                        "pruning_estimations": False,
                        "quantized_estimations": False,
                        "iterations_per_check": 0,
                        "warmup_iterations_per_check": 0,
                    }
                ]
            },
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": None,
                        "job": None,
                        "name": None,
                        "analysis": None,
                        "batch_size": None,
                        "core_count": None,
                        "instruction_sets": None,
                        "pruning_estimations": False,
                        "quantized_estimations": False,
                        "iterations_per_check": 0,
                        "warmup_iterations_per_check": 0,
                    }
                ]
            },
            None,
        ),
        (
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": "uploaded",
                        "job": None,
                        "name": "profile",
                        "batch_size": 64,
                        "core_count": 4,
                        "instruction_sets": ["AVX2", "AVX512", "VNNI"],
                        "pruning_estimations": True,
                        "quantized_estimations": True,
                        "iterations_per_check": 30,
                        "warmup_iterations_per_check": 5,
                        "analysis": None,
                    }
                ]
            },
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": "uploaded",
                        "job": None,
                        "name": "profile",
                        "batch_size": 64,
                        "core_count": 4,
                        "instruction_sets": ["AVX2", "AVX512", "VNNI"],
                        "pruning_estimations": True,
                        "quantized_estimations": True,
                        "iterations_per_check": 30,
                        "warmup_iterations_per_check": 5,
                        "analysis": None,
                    }
                ]
            },
            None,
        ),
        (
            {
                "profiles": [
                    {
                        "profile_id": "profile id",
                        "project_id": "project id",
                        "source": "fail",
                        "job": None,
                        "name": "profile",
                        "analysis": None,
                        "batch_size": 64,
                        "core_count": 4,
                        "instruction_sets": "fail",
                        "pruning_estimations": True,
                        "quantized_estimations": True,
                        "iterations_per_check": 30,
                        "warmup_iterations_per_check": 5,
                    }
                ]
            },
            None,
            ["profiles"],
        ),
    ],
)
def test_repsonse_project_perf_profiles_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["profiles"][0]["created"] = created
    if expected_output:
        expected_output["profiles"][0]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectPerfProfilesSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"profile_id": "profile id", "project_id": "project id"},
            {"profile_id": "profile id", "project_id": "project id", "success": True},
        ),
        (
            {"profile_id": "profile id", "project_id": "project id", "success": False},
            {"profile_id": "profile id", "project_id": "project id", "success": False},
        ),
    ],
)
def response_project_profile_deleted_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ResponseProjectProfileDeletedSchema,
        expected_input,
        expected_output,
    )
