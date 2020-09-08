from typing import Dict, List, Union
import pytest
from datetime import datetime
from neuralmagicML.server.schemas.projects_data import (
    ProjectDataSchema,
    ResponseProjectDataSchema,
    ResponseProjectDataDeletedSchema,
    SetProjectDataFromSchema,
)

from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "file": None,
            },
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "file": None,
            },
            None,
        ),
        (
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": "downloaded_path",
                "job": None,
                "file": "/path/to/file",
            },
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": "downloaded_path",
                "job": None,
                "file": "/path/to/file",
            },
            None,
        ),
        (
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": "fail",
                "job": None,
                "file": "/path/to/file",
            },
            None,
            ["source"],
        ),
    ],
)
def test_project_data_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["created"] = created.isoformat()
    schema_tester(
        ProjectDataSchema, expected_input, expected_output, expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": None,
                "job": None,
                "file": None,
            },
            {
                "data": [
                    {
                        "data_id": "data id",
                        "project_id": "project id",
                        "source": None,
                        "job": None,
                        "file": None,
                    }
                ]
            },
            None,
        ),
        (
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": "downloaded_path",
                "job": None,
                "file": "/path/to/file",
            },
            {
                "data": [
                    {
                        "data_id": "data id",
                        "project_id": "project id",
                        "source": "downloaded_path",
                        "job": None,
                        "file": "/path/to/file",
                    }
                ]
            },
            None,
        ),
        (
            {
                "data_id": "data id",
                "project_id": "project id",
                "source": "fail",
                "job": None,
                "file": "/path/to/file",
            },
            None,
            ["data"],
        ),
    ],
)
def test_response_project_data_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["data"][0]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectDataSchema,
        {"data": [expected_input]},
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"project_id": "project id", "data_id": "data id"},
            {"success": True, "project_id": "project id", "data_id": "data id"},
        ),
        (
            {"success": False, "project_id": "project id", "data_id": "data id"},
            {"success": False, "project_id": "project id", "data_id": "data id"},
        ),
    ],
)
def test_response_project_data_Deleted_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ResponseProjectDataDeletedSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output", [({"uri": "path"}, {"uri": "path"})]
)
def test_set_project_data_from_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(SetProjectDataFromSchema, expected_input, expected_output)
