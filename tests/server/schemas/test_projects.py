from typing import Dict, List, Union
from datetime import datetime
import pytest
from neuralmagicML.server.schemas.projects import (
    ProjectSchema,
    ProjectExtSchema,
    ResponseProjectSchema,
    ResponseProjectExtSchema,
    ResponseProjectsSchema,
    ResponseProjectDeletedSchema,
    SearchProjectsSchema,
    CreateUpdateProjectSchema,
    DeleteProjectSchema,
)

from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
        ),
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
        ),
    ],
)
def test_project_schema(expected_input: Dict, expected_output: Dict):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_output["created"] = created.isoformat()
    expected_input["modified"] = modified
    expected_output["modified"] = modified.isoformat()
    schema_tester(ProjectSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
                "data": None,
                "model": None,
            },
        ),
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
                "data": [
                    {
                        "data_id": "data id",
                        "project_id": "project id",
                        "source": "downloaded_path",
                        "job": None,
                        "file": "/path/to/file",
                    }
                ],
                "model": {
                    "model_id": "model id",
                    "project_id": "project id",
                    "source": "downloaded_path",
                    "job": None,
                    "file": "/path/to/file",
                    "analysis": None,
                },
            },
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
                "data": [
                    {
                        "data_id": "data id",
                        "project_id": "project id",
                        "source": "downloaded_path",
                        "job": None,
                        "file": "/path/to/file",
                    }
                ],
                "model": {
                    "model_id": "model id",
                    "project_id": "project id",
                    "source": "downloaded_path",
                    "job": None,
                    "file": "/path/to/file",
                    "analysis": None,
                },
            },
        ),
    ],
)
def test_project_ext_schema(expected_input: Dict, expected_output: Dict):
    created = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = created
    expected_output["created"] = created.isoformat()
    expected_output["modified"] = created.isoformat()
    if "model" in expected_input and expected_input["model"]:
        expected_input["model"]["created"] = created
    if "data" in expected_input and expected_input["data"]:
        expected_input["data"][0]["created"] = created
    if expected_output["model"]:
        expected_output["model"]["created"] = created.isoformat()
    if expected_output["data"]:
        expected_output["data"][0]["created"] = created.isoformat()
    schema_tester(ProjectExtSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "project": {
                    "project_id": "project_id",
                    "name": "project name",
                    "description": "description",
                    "training_optimizer": None,
                    "training_epochs": None,
                    "training_lr_init": None,
                    "training_lr_final": None,
                    "dir_path": "path/to/file",
                    "dir_size": 10000,
                }
            },
        ),
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "project": {
                    "project_id": "project_id",
                    "name": "project name",
                    "description": "description",
                    "training_optimizer": "optimizer",
                    "training_epochs": 30,
                    "training_lr_init": 0.005,
                    "training_lr_final": 0.0001,
                    "dir_path": "path/to/file",
                    "dir_size": 10000,
                }
            },
        ),
    ],
)
def test_response_project_schema(expected_input: Dict, expected_output: Dict):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_output["project"]["created"] = created.isoformat()
    expected_input["modified"] = modified
    expected_output["project"]["modified"] = modified.isoformat()
    schema_tester(ResponseProjectSchema, {"project": expected_input}, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "project": {
                    "project_id": "project_id",
                    "name": "project name",
                    "description": "description",
                    "training_optimizer": None,
                    "training_epochs": None,
                    "training_lr_init": None,
                    "training_lr_final": None,
                    "dir_path": "path/to/file",
                    "dir_size": 10000,
                    "data": None,
                    "model": None,
                }
            },
        ),
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
                "data": [
                    {
                        "data_id": "data id",
                        "project_id": "project id",
                        "source": "downloaded_path",
                        "job": None,
                        "file": "/path/to/file",
                    }
                ],
                "model": {
                    "model_id": "model id",
                    "project_id": "project id",
                    "source": "downloaded_path",
                    "job": None,
                    "file": "/path/to/file",
                    "analysis": None,
                },
            },
            {
                "project": {
                    "project_id": "project_id",
                    "name": "project name",
                    "description": "description",
                    "training_optimizer": "optimizer",
                    "training_epochs": 30,
                    "training_lr_init": 0.005,
                    "training_lr_final": 0.0001,
                    "dir_path": "path/to/file",
                    "dir_size": 10000,
                    "data": [
                        {
                            "data_id": "data id",
                            "project_id": "project id",
                            "source": "downloaded_path",
                            "job": None,
                            "file": "/path/to/file",
                        }
                    ],
                    "model": {
                        "model_id": "model id",
                        "project_id": "project id",
                        "source": "downloaded_path",
                        "job": None,
                        "file": "/path/to/file",
                        "analysis": None,
                    },
                }
            },
        ),
    ],
)
def test_project_ext_schema(expected_input: Dict, expected_output: Dict):
    created = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = created
    expected_output["project"]["created"] = created.isoformat()
    expected_output["project"]["modified"] = created.isoformat()
    if "model" in expected_input and expected_input["model"]:
        expected_input["model"]["created"] = created
    if "data" in expected_input and expected_input["data"]:
        expected_input["data"][0]["created"] = created
    if expected_output["project"]["model"]:
        expected_output["project"]["model"]["created"] = created.isoformat()
    if expected_output["project"]["data"]:
        expected_output["project"]["data"][0]["created"] = created.isoformat()
    schema_tester(
        ResponseProjectExtSchema, {"project": expected_input}, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "projects": [
                    {
                        "project_id": "project_id",
                        "name": "project name",
                        "description": "description",
                        "training_optimizer": None,
                        "training_epochs": None,
                        "training_lr_init": None,
                        "training_lr_final": None,
                        "dir_path": "path/to/file",
                        "dir_size": 10000,
                    }
                ]
            },
        ),
        (
            {
                "project_id": "project_id",
                "name": "project name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 30,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0001,
                "dir_path": "path/to/file",
                "dir_size": 10000,
            },
            {
                "projects": [
                    {
                        "project_id": "project_id",
                        "name": "project name",
                        "description": "description",
                        "training_optimizer": "optimizer",
                        "training_epochs": 30,
                        "training_lr_init": 0.005,
                        "training_lr_final": 0.0001,
                        "dir_path": "path/to/file",
                        "dir_size": 10000,
                    }
                ]
            },
        ),
    ],
)
def test_response_projects_schema(expected_input: Dict, expected_output: Dict):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_output["projects"][0]["created"] = created.isoformat()
    expected_input["modified"] = modified
    expected_output["projects"][0]["modified"] = modified.isoformat()
    schema_tester(
        ResponseProjectsSchema, {"projects": [expected_input]}, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({"project_id": "project_id"}, {"success": True, "project_id": "project_id"}),
        (
            {"project_id": "project_id", "success": False},
            {"project_id": "project_id", "success": False},
        ),
    ],
)
def test_response_project_deleted_schema(expected_input: Dict, expected_output: Dict):
    schema_tester(ResponseProjectDeletedSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {},
            {"order_by": "modified", "order_desc": True, "page": 1, "page_length": 20},
            None,
        ),
        (
            {"order_by": "name", "order_desc": False, "page": 5, "page_length": 15},
            {"order_by": "name", "order_desc": False, "page": 5, "page_length": 15},
            None,
        ),
        (
            {"order_by": "fail", "order_desc": False, "page": 1, "page_length": 20},
            None,
            ["order_by"],
        ),
        (
            {"order_by": "created", "order_desc": False, "page": 0, "page_length": 20},
            None,
            ["page"],
        ),
        (
            {"order_by": "created", "order_desc": False, "page": 1, "page_length": 0},
            None,
            ["page_length"],
        ),
    ],
)
def test_search_projects_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        SearchProjectsSchema, expected_input, expected_output, expect_validation_error
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {}),
        (
            {
                "name": "name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
            },
            {
                "name": "name",
                "description": "description",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
            },
        ),
        (
            {
                "name": "name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 1,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0005,
            },
            {
                "name": "name",
                "description": "description",
                "training_optimizer": "optimizer",
                "training_epochs": 1,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0005,
            },
        ),
    ],
)
def test_create_update_project_schema(expected_input: Dict, expected_output: Dict):
    schema_tester(CreateUpdateProjectSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [({}, {"force": False}), ({"force": True}, {"force": True})],
)
def test_delete_project_schema(expected_input: Dict, expected_output: Dict):
    schema_tester(DeleteProjectSchema, expected_input, expected_output)
