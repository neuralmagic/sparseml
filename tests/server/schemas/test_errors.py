from typing import Dict
import pytest
from tests.server.helper import schema_tester
from sparseml.server.schemas.errors import ErrorSchema


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"error_type": "ValidationError", "error_message": "Some validation error"},
            {
                "success": False,
                "error_code": -1,
                "error_type": "ValidationError",
                "error_message": "Some validation error",
            },
        ),
        (
            {
                "success": True,
                "error_code": 404,
                "error_type": "ValidationError",
                "error_message": "Some validation error",
            },
            {
                "success": True,
                "error_code": 404,
                "error_type": "ValidationError",
                "error_message": "Some validation error",
            },
        ),
    ],
)
def test_error_schema(expected_input: Dict, expected_output: Dict):
    schema_tester(ErrorSchema, expected_input, expected_output)
