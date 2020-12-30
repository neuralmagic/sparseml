from typing import Dict, List, Union
import pytest
from datetime import datetime
from sparseml.server.schemas.projects_data import (
    ProjectDataSchema,
    SearchProjectDataSchema,
    CreateUpdateProjectDataSchema,
    ResponseProjectDataSingleSchema,
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
        ProjectDataSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "source": None,
                "job": None,
                "file": None,
            },
            {
                "source": None,
                "job": None,
                "file": None,
            },
            None,
        ),
        (
            {
                "source": "downloaded_path",
                "job": None,
                "file": "/path/to/file",
            },
            {
                "source": "downloaded_path",
                "job": None,
                "file": "/path/to/file",
            },
            None,
        ),
        (
            {
                "source": "fail",
                "job": None,
                "file": "/path/to/file",
            },
            None,
            ["source"],
        ),
    ],
)
def test_create_update_project_data_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        CreateUpdateProjectDataSchema,
        expected_input,
        expected_output,
        expect_validation_error,
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
            {"page": 5, "page_length": 15},
            {"page": 5, "page_length": 15},
            None,
        ),
        (
            {"page": 0, "page_length": 20},
            None,
            ["page"],
        ),
        (
            {"page": 1, "page_length": 0},
            None,
            ["page_length"],
        ),
    ],
)
def test_search_project_data_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        SearchProjectDataSchema,
        expected_input,
        expected_output,
        expect_validation_error,
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
                "data": {
                    "data_id": "data id",
                    "project_id": "project id",
                    "source": None,
                    "job": None,
                    "file": None,
                }
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
                "data": {
                    "data_id": "data id",
                    "project_id": "project id",
                    "source": "downloaded_path",
                    "job": None,
                    "file": "/path/to/file",
                }
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
def test_response_project_singles_data_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    expected_input["created"] = created
    if expected_output:
        expected_output["data"]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectDataSingleSchema,
        {"data": expected_input},
        expected_output,
        expect_validation_error,
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
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(ResponseProjectDataDeletedSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output", [({"uri": "path"}, {"uri": "path"})]
)
def test_set_project_data_from_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(SetProjectDataFromSchema, expected_input, expected_output)
