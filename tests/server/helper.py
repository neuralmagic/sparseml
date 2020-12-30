import os
import pytest
from typing import Dict, List, Union
from marshmallow import Schema
from sparseml.server.models import (
    database_setup,
)

TEMP_DIR = os.path.expanduser("~/.cache/nm_database")

__all__ = ["schema_tester", "database_fixture"]


@pytest.fixture(scope="session")
def database_fixture():
    os.makedirs(TEMP_DIR, exist_ok=True)
    database_setup(TEMP_DIR)
    yield
    if os.path.exists(os.path.join(TEMP_DIR, "db.sqlite")):
        os.remove(os.path.join(TEMP_DIR, "db.sqlite"))


def schema_tester(
    schema_constr: Schema,
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None] = None,
):
    schema = schema_constr()
    formatted = schema.dump(expected_input)
    errors = schema.validate(formatted)
    if expect_validation_error is not None:
        assert set(errors.keys()) == set(expect_validation_error)

    else:
        assert errors == {}
        assert formatted == expected_output
