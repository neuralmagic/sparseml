from typing import Dict, List, Union
import pytest
from datetime import datetime
from neuralmagicML.server.schemas.projects_model import (
    ProjectModelSchema,
    ProjectModelAnalysisSchema,
    CreateUpdateProjectModelSchema,
    ResponseProjectModelAnalysisSchema,
    ResponseProjectModelSchema,
    ResponseProjectModelDeletedSchema,
    SetProjectModelFromSchema,
    DeleteProjectModelSchema,
)

from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "nodes": [
                    {
                        "id": "id",
                        "op_type": "Conv",
                        "input_names": ["input_1"],
                        "output_names": ["output_1"],
                        "input_shapes": None,
                        "output_shapes": None,
                        "params": 0,
                        "prunable": False,
                        "prunable_params": -1,
                        "prunable_params_zeroed": 0,
                        "flops": None,
                        "weight_name": None,
                        "weight_shape": None,
                        "bias_name": None,
                        "bias_shape": None,
                        "attributes": None,
                    }
                ]
            },
            {
                "nodes": [
                    {
                        "id": "id",
                        "op_type": "Conv",
                        "input_names": ["input_1"],
                        "output_names": ["output_1"],
                        "input_shapes": None,
                        "output_shapes": None,
                        "params": 0,
                        "prunable": False,
                        "prunable_params": -1,
                        "prunable_params_zeroed": 0,
                        "flops": None,
                        "weight_name": None,
                        "weight_shape": None,
                        "bias_name": None,
                        "bias_shape": None,
                        "attributes": None,
                    }
                ]
            },
        ),
        (
            {
                "nodes": [
                    {
                        "attributes": {
                            "dilations": [1, 1],
                            "group": 1,
                            "kernel_shape": [3, 3],
                            "pads": [1, 1, 1, 1],
                            "strides": [1, 1],
                        },
                        "bias_name": "sections.0.0.conv.bias",
                        "bias_shape": [64],
                        "flops": 89915392,
                        "id": "39",
                        "input_names": ["input"],
                        "input_shapes": [[1, 3, 224, 224]],
                        "op_type": "Conv",
                        "output_names": ["39"],
                        "output_shapes": [[1, 64, 224, 224]],
                        "params": 1792,
                        "prunable": True,
                        "prunable_params": 1728,
                        "prunable_params_zeroed": 0,
                        "weight_name": "sections.0.0.conv.weight",
                        "weight_shape": [64, 3, 3, 3],
                    },
                    {
                        "attributes": {},
                        "bias_name": None,
                        "bias_shape": None,
                        "flops": 3211264,
                        "id": "40",
                        "input_names": ["39"],
                        "input_shapes": [[1, 64, 224, 224]],
                        "op_type": "Relu",
                        "output_names": ["40"],
                        "output_shapes": [[1, 64, 224, 224]],
                        "params": 0,
                        "prunable": False,
                        "prunable_params": -1,
                        "prunable_params_zeroed": 0,
                        "weight_name": None,
                        "weight_shape": None,
                    },
                ]
            },
            {
                "nodes": [
                    {
                        "attributes": {
                            "dilations": [1, 1],
                            "group": 1,
                            "kernel_shape": [3, 3],
                            "pads": [1, 1, 1, 1],
                            "strides": [1, 1],
                        },
                        "bias_name": "sections.0.0.conv.bias",
                        "bias_shape": [64],
                        "flops": 89915392,
                        "id": "39",
                        "input_names": ["input"],
                        "input_shapes": [[1, 3, 224, 224]],
                        "op_type": "Conv",
                        "output_names": ["39"],
                        "output_shapes": [[1, 64, 224, 224]],
                        "params": 1792,
                        "prunable": True,
                        "prunable_params": 1728,
                        "prunable_params_zeroed": 0,
                        "weight_name": "sections.0.0.conv.weight",
                        "weight_shape": [64, 3, 3, 3],
                    },
                    {
                        "attributes": {},
                        "bias_name": None,
                        "bias_shape": None,
                        "flops": 3211264,
                        "id": "40",
                        "input_names": ["39"],
                        "input_shapes": [[1, 64, 224, 224]],
                        "op_type": "Relu",
                        "output_names": ["40"],
                        "output_shapes": [[1, 64, 224, 224]],
                        "params": 0,
                        "prunable": False,
                        "prunable_params": -1,
                        "prunable_params_zeroed": 0,
                        "weight_name": None,
                        "weight_shape": None,
                    },
                ]
            },
        ),
    ],
)
def test_project_model_analysis_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectModelAnalysisSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "model_id": "model id",
                "project_id": "project_id",
                "source": "downloaded_repo",
                "job": None,
                "file": "/path/to/file",
                "analysis": None,
            },
            {
                "model_id": "model id",
                "project_id": "project_id",
                "source": "downloaded_repo",
                "job": None,
                "file": "/path/to/file",
                "analysis": None,
            },
            None,
        ),
        (
            {
                "model_id": "model id",
                "project_id": "project_id",
                "source": "uploaded",
                "job": None,
                "file": "/path/to/file",
                "analysis": {
                    "nodes": [
                        {
                            "attributes": {
                                "dilations": [1, 1],
                                "group": 1,
                                "kernel_shape": [3, 3],
                                "pads": [1, 1, 1, 1],
                                "strides": [1, 1],
                            },
                            "bias_name": "sections.0.0.conv.bias",
                            "bias_shape": [64],
                            "flops": 89915392,
                            "id": "39",
                            "input_names": ["input"],
                            "input_shapes": [[1, 3, 224, 224]],
                            "op_type": "Conv",
                            "output_names": ["39"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 1792,
                            "prunable": True,
                            "prunable_params": 1728,
                            "prunable_params_zeroed": 0,
                            "weight_name": "sections.0.0.conv.weight",
                            "weight_shape": [64, 3, 3, 3],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 3211264,
                            "id": "40",
                            "input_names": ["39"],
                            "input_shapes": [[1, 64, 224, 224]],
                            "op_type": "Relu",
                            "output_names": ["40"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 0,
                            "prunable": False,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                },
            },
            {
                "model_id": "model id",
                "project_id": "project_id",
                "source": "uploaded",
                "job": None,
                "file": "/path/to/file",
                "analysis": {
                    "nodes": [
                        {
                            "attributes": {
                                "dilations": [1, 1],
                                "group": 1,
                                "kernel_shape": [3, 3],
                                "pads": [1, 1, 1, 1],
                                "strides": [1, 1],
                            },
                            "bias_name": "sections.0.0.conv.bias",
                            "bias_shape": [64],
                            "flops": 89915392,
                            "id": "39",
                            "input_names": ["input"],
                            "input_shapes": [[1, 3, 224, 224]],
                            "op_type": "Conv",
                            "output_names": ["39"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 1792,
                            "prunable": True,
                            "prunable_params": 1728,
                            "prunable_params_zeroed": 0,
                            "weight_name": "sections.0.0.conv.weight",
                            "weight_shape": [64, 3, 3, 3],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 3211264,
                            "id": "40",
                            "input_names": ["39"],
                            "input_shapes": [[1, 64, 224, 224]],
                            "op_type": "Relu",
                            "output_names": ["40"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 0,
                            "prunable": False,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                },
            },
            None,
        ),
    ],
)
def test_project_model_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["created"] = created.isoformat()
    schema_tester(
        ProjectModelSchema, expected_input, expected_output, expect_validation_error
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {"file": "/path/to/file", "source": "uploaded", "job": None},
            {"file": "/path/to/file", "source": "uploaded", "job": None},
            None
        ),
        ({"job": None}, {"job": None}, None),
        ({"job": None, "source": "failure"}, None, ["source"]),
    ],
)
def test_create_update_project_model_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        CreateUpdateProjectModelSchema,
        expected_input,
        expected_output,
        expect_validation_error
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "analysis": {
                    "nodes": [
                        {
                            "id": "id",
                            "op_type": "Conv",
                            "input_names": ["input_1"],
                            "output_names": ["output_1"],
                            "input_shapes": None,
                            "output_shapes": None,
                            "params": 0,
                            "prunable": False,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "flops": None,
                            "weight_name": None,
                            "weight_shape": None,
                            "bias_name": None,
                            "bias_shape": None,
                            "attributes": None,
                        }
                    ]
                }
            },
            {
                "analysis": {
                    "nodes": [
                        {
                            "id": "id",
                            "op_type": "Conv",
                            "input_names": ["input_1"],
                            "output_names": ["output_1"],
                            "input_shapes": None,
                            "output_shapes": None,
                            "params": 0,
                            "prunable": False,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "flops": None,
                            "weight_name": None,
                            "weight_shape": None,
                            "bias_name": None,
                            "bias_shape": None,
                            "attributes": None,
                        }
                    ]
                },
            },
        ),
        (
            {
                "analysis": {
                    "nodes": [
                        {
                            "attributes": {
                                "dilations": [1, 1],
                                "group": 1,
                                "kernel_shape": [3, 3],
                                "pads": [1, 1, 1, 1],
                                "strides": [1, 1],
                            },
                            "bias_name": "sections.0.0.conv.bias",
                            "bias_shape": [64],
                            "flops": 89915392,
                            "id": "39",
                            "input_names": ["input"],
                            "input_shapes": [[1, 3, 224, 224]],
                            "op_type": "Conv",
                            "output_names": ["39"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 1792,
                            "prunable": True,
                            "prunable_params": 1728,
                            "prunable_params_zeroed": 0,
                            "weight_name": "sections.0.0.conv.weight",
                            "weight_shape": [64, 3, 3, 3],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 3211264,
                            "id": "40",
                            "input_names": ["39"],
                            "input_shapes": [[1, 64, 224, 224]],
                            "op_type": "Relu",
                            "output_names": ["40"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 0,
                            "prunable": False,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                }
            },
            {
                "analysis": {
                    "nodes": [
                        {
                            "attributes": {
                                "dilations": [1, 1],
                                "group": 1,
                                "kernel_shape": [3, 3],
                                "pads": [1, 1, 1, 1],
                                "strides": [1, 1],
                            },
                            "bias_name": "sections.0.0.conv.bias",
                            "bias_shape": [64],
                            "flops": 89915392,
                            "id": "39",
                            "input_names": ["input"],
                            "input_shapes": [[1, 3, 224, 224]],
                            "op_type": "Conv",
                            "output_names": ["39"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 1792,
                            "prunable": True,
                            "prunable_params": 1728,
                            "prunable_params_zeroed": 0,
                            "weight_name": "sections.0.0.conv.weight",
                            "weight_shape": [64, 3, 3, 3],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 3211264,
                            "id": "40",
                            "input_names": ["39"],
                            "input_shapes": [[1, 64, 224, 224]],
                            "op_type": "Relu",
                            "output_names": ["40"],
                            "output_shapes": [[1, 64, 224, 224]],
                            "params": 0,
                            "prunable": False,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                }
            },
        ),
    ],
)
def test_response_project_model_analysis_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ResponseProjectModelAnalysisSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"project_id": "project id", "model_id": "model id"},
            {"success": True, "project_id": "project id", "model_id": "model id"},
        ),
        (
            {"success": False, "project_id": "project id", "model_id": "model id"},
            {"success": False, "project_id": "project id", "model_id": "model id"},
        ),
    ],
)
def test_response_project_model_deleted_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ResponseProjectModelDeletedSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({"uri": "https://location.com"}, {"uri": "https://location.com"}),
    ],
)
def test_set_project_model_from_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        SetProjectModelFromSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {"force": False}),
        (
            {
                "force": True,
            },
            {"force": True},
        ),
    ],
)
def test_delete_project_model_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        DeleteProjectModelSchema,
        expected_input,
        expected_output,
    )
