from typing import Dict, List, Union
from marshmallow import Schema

__all__ = ["schema_tester"]


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
