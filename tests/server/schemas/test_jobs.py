from typing import Dict, List, Union
from datetime import datetime
import pytest
from neuralmagicML.server.schemas.jobs import (
    JobProgressSchema,
    JobSchema,
    ResponseJobSchema,
    ResponseJobsSchema,
    SearchJobsSchema,
)
from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {"iter_indefinite": False, "iter_class": "class"},
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": None,
                "num_steps": 1,
                "step_class": None,
                "step_index": 0,
            },
            None,
        ),
        (
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 0.5,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": 10,
            },
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 0.5,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": 10,
            },
            None,
        ),
        (
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": -1,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": 10,
            },
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": -1,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": 10,
            },
            ["iter_val"],
        ),
        (
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 10,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": 10,
            },
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 10,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": 10,
            },
            ["iter_val"],
        ),
        (
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 0.5,
                "num_steps": 0,
                "step_class": "Step class",
                "step_index": 10,
            },
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 0.5,
                "num_steps": 0,
                "step_class": "Step class",
                "step_index": 10,
            },
            ["num_steps"],
        ),
        (
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 0.5,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": -1,
            },
            {
                "iter_indefinite": False,
                "iter_class": "class",
                "iter_val": 0.5,
                "num_steps": 100,
                "step_class": "Step class",
                "step_index": -1,
            },
            ["step_index"],
        ),
    ],
)
def test_job_progress_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        JobProgressSchema, expected_input, expected_output, expect_validation_error
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": None,
                "status": "pending",
                "progress": None,
                "error": None,
            },
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": None,
                "status": "pending",
                "progress": None,
                "error": None,
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "completed",
                "progress": {
                    "iter_indefinite": False,
                    "iter_class": "iter class",
                    "iter_val": 1,
                },
                "error": None,
            },
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "completed",
                "progress": {
                    "step_index": 0,
                    "num_steps": 1,
                    "iter_val": 1.0,
                    "iter_class": "iter class",
                    "step_class": None,
                    "iter_indefinite": False,
                },
                "error": None,
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "canceled",
                "progress": None,
                "error": "canceled",
            },
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "canceled",
                "progress": None,
                "error": "canceled",
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "unknown",
                "progress": None,
                "error": "canceled",
            },
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "unknown",
                "progress": None,
                "error": "canceled",
            },
            ["status"],
        ),
    ],
)
def test_job_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_output["created"] = created.isoformat()
    expected_input["modified"] = modified
    expected_output["modified"] = modified.isoformat()
    schema_tester(JobSchema, expected_input, expected_output, expect_validation_error)


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": None,
                "status": "pending",
                "progress": None,
                "error": None,
            },
            {
                "job": {
                    "job_id": "id",
                    "project_id": "project_id",
                    "type_": "job type",
                    "worker_args": None,
                    "status": "pending",
                    "progress": None,
                    "error": None,
                }
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "completed",
                "progress": {
                    "iter_indefinite": False,
                    "iter_class": "iter class",
                    "iter_val": 1,
                },
                "error": None,
            },
            {
                "job": {
                    "job_id": "id",
                    "project_id": "project_id",
                    "type_": "job type",
                    "worker_args": {"arg1": "val1", "arg2": "val2"},
                    "status": "completed",
                    "progress": {
                        "step_index": 0,
                        "num_steps": 1,
                        "iter_val": 1.0,
                        "iter_class": "iter class",
                        "step_class": None,
                        "iter_indefinite": False,
                    },
                    "error": None,
                }
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "canceled",
                "progress": None,
                "error": "canceled",
            },
            {
                "job": {
                    "job_id": "id",
                    "project_id": "project_id",
                    "type_": "job type",
                    "worker_args": {"arg1": "val1", "arg2": "val2"},
                    "status": "canceled",
                    "progress": None,
                    "error": "canceled",
                }
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "unknown",
                "progress": None,
                "error": "canceled",
            },
            {
                "job": {
                    "job_id": "id",
                    "project_id": "project_id",
                    "type_": "job type",
                    "worker_args": {"arg1": "val1", "arg2": "val2"},
                    "status": "unknown",
                    "progress": None,
                    "error": "canceled",
                }
            },
            ["job"],
        ),
    ],
)
def test_response_job_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_output["job"]["created"] = created.isoformat()
    expected_input["modified"] = modified
    expected_output["job"]["modified"] = modified.isoformat()
    schema_tester(
        ResponseJobSchema,
        {"job": expected_input},
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": None,
                "status": "pending",
                "progress": None,
                "error": None,
            },
            {
                "jobs": [
                    {
                        "job_id": "id",
                        "project_id": "project_id",
                        "type_": "job type",
                        "worker_args": None,
                        "status": "pending",
                        "progress": None,
                        "error": None,
                    }
                ]
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "completed",
                "progress": {
                    "iter_indefinite": False,
                    "iter_class": "iter class",
                    "iter_val": 1,
                },
                "error": None,
            },
            {
                "jobs": [
                    {
                        "job_id": "id",
                        "project_id": "project_id",
                        "type_": "job type",
                        "worker_args": {"arg1": "val1", "arg2": "val2"},
                        "status": "completed",
                        "progress": {
                            "step_index": 0,
                            "num_steps": 1,
                            "iter_val": 1.0,
                            "iter_class": "iter class",
                            "step_class": None,
                            "iter_indefinite": False,
                        },
                        "error": None,
                    }
                ]
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "canceled",
                "progress": None,
                "error": "canceled",
            },
            {
                "jobs": [
                    {
                        "job_id": "id",
                        "project_id": "project_id",
                        "type_": "job type",
                        "worker_args": {"arg1": "val1", "arg2": "val2"},
                        "status": "canceled",
                        "progress": None,
                        "error": "canceled",
                    }
                ]
            },
            None,
        ),
        (
            {
                "job_id": "id",
                "project_id": "project_id",
                "type_": "job type",
                "worker_args": {"arg1": "val1", "arg2": "val2"},
                "status": "unknown",
                "progress": None,
                "error": "canceled",
            },
            {
                "jobs": [
                    {
                        "job_id": "id",
                        "project_id": "project_id",
                        "type_": "job type",
                        "worker_args": {"arg1": "val1", "arg2": "val2"},
                        "status": "unknown",
                        "progress": None,
                        "error": "canceled",
                    }
                ]
            },
            ["jobs"],
        ),
    ],
)
def test_response_jobs_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_output["jobs"][0]["created"] = created.isoformat()
    expected_input["modified"] = modified
    expected_output["jobs"][0]["modified"] = modified.isoformat()
    schema_tester(
        ResponseJobsSchema,
        {"jobs": [expected_input]},
        expected_output,
        expect_validation_error,
    )
