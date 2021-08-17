# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions for base Modifier and Manger utilities
"""


import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import yaml


__all__ = [
    "VALID_RECIPE_METADATA_FIELDS",
    "RecipeMetadataField",
    "maybe_evaluate_recipe_equation",
    "maybe_evaluate_yaml_object",
    "evaluate_recipe_variables",
    "evaluate_recipe_yaml_str_equations",
]


@dataclass
class RecipeMetadataField:
    """
    Information about a valid SparseML recipe metadata field

    :param valid_types: tuple of valid types that the given parameter can be
    :param can_be_variable: True if the field can be referenced as a variable
        in the recipe, False otherwise
    """

    valid_types: Tuple
    can_be_variable: bool


VALID_RECIPE_METADATA_FIELDS = {
    "num_epochs": RecipeMetadataField((int, float), True),
    "init_lr": RecipeMetadataField((float,), True),
}


def validate_recipe_metadata(
    metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Union[int, float]]]:
    """
    Validates the recipe metadata field of a recipe that all key and value types
    are expected

    :param metadata: metadata Dict to be validated
    :return: tuple of the validated metadata Dict and sub-dictionary of the valid
        metadata fields that can be used as variables in the recipe
    """
    if not isinstance(metadata, Dict):
        raise ValueError(
            f"Recipe metadata must be a Dict, received metadata type {type(metadata)}"
        )

    metadata_vars = {}
    for name, val in metadata.items():
        if not isinstance(name, str):
            raise ValueError(
                f"metadata key must be a str. received key {name} of type {type(name)}"
            )

        if name not in VALID_RECIPE_METADATA_FIELDS:
            raise ValueError(
                f"Invalid recipe metadata field: {name}. "
                f"Valid fields: {list(VALID_RECIPE_METADATA_FIELDS.keys())}"
            )
        expected_types = VALID_RECIPE_METADATA_FIELDS[name].valid_types
        if not isinstance(val, expected_types):
            raise ValueError(
                f"Expected metadata field: {name} to have type in: {expected_types}. "
                f"Received object of type {type(val)}"
            )

        if VALID_RECIPE_METADATA_FIELDS[name].can_be_variable:
            metadata_vars[name] = val

    return metadata, metadata_vars


def maybe_evaluate_recipe_equation(
    val: str,
    variables: Dict[str, Union[int, float]],
) -> Union[str, float, int]:
    """
    :param val: string to evaluate
    :param variables: dictionary of possible variable names to the numerical
        values they hold
    :return: if the given string consists of only numbers, variable names that
        are defined in `variables`, or any operator in ['+', '-', '*', or '/']
        then the numeric result of that expression will be returned. Otherwise,
        the same string will be returned. if the value string is wrapped
        with `eval(...)` and the value fails to evaluate based on valid values
        (numbers, metadata variable names, and arithmetic signs) an error will
        be raised.
    """
    valid_operations = {"+", "-", "*", "/"}

    if val.startswith("eval(") and val.endswith(")"):
        is_eval_str = True
        val = val[5:-1]
    else:
        is_eval_str = False

    equation = val.split()  # split on whitespace

    for idx, part in enumerate(equation):
        part = _maybe_parse_number(part)  # attempt to parse number from string
        part = variables.get(part, part)  # attempt to load from variable

        if isinstance(part, str) and part not in valid_operations:
            # not a number, valid variable, or valid operation. return unchanged val
            if is_eval_str:
                _raise_invalid_eval_exception(val)
            return val

        # assert that if part this is added to the final equation,
        # it will be a number or operation
        assert part in valid_operations or isinstance(part, (int, float))

        equation[idx] = str(part)

    try:
        evaluated_val = eval(" ".join(equation))
    except Exception:
        evaluated_val = val

    if is_eval_str and not isinstance(evaluated_val, (int, float)):
        _raise_invalid_eval_exception(evaluated_val)

    return evaluated_val


def maybe_evaluate_yaml_object(
    obj: Any, variables: Dict[str, Union[int, float]]
) -> Any:
    """

    :param obj: object to evaluate string elements of. will nest through
        any lists and dictionary values. Any object that is not a string,
        list, or dictionary will be unchanged
    :param variables: dictionary of possible variable names to the numerical
        values they hold
    :return: the object with any valid evaluations mde
    """
    if isinstance(obj, str):
        return maybe_evaluate_recipe_equation(obj, variables)
    elif isinstance(obj, list):
        return [maybe_evaluate_yaml_object(val, variables) for val in obj]
    elif isinstance(obj, dict):
        return {
            key: maybe_evaluate_yaml_object(val, variables) for key, val in obj.items()
        }
    else:
        return obj


def evaluate_recipe_variables(
    recipe_dict: Dict[str, Any],
    metadata_variables: Dict[str, Union[int, float]],
) -> Tuple[Dict[str, Any], Dict[str, Union[int, float]]]:
    """
    evaluates any string values in the recipe dictionary based on metadata variables
    and simple arithmetic and sets them in the dictionary. if a variable string
    is wrapped with `eval(...)` and the value fails to evaluate based on valid
    values (numbers, metadata variable names, and arithmetic signs) an error will
    be raised

    :param recipe_dict: dictionary of values from a loaded base recipe string
    :param metadata_variables: dictionary of loaded metadata variables from
        validate_recipe_metadata
    :return: the processed variables dictionary (including metadata variables)
        with any string values evaluated based on the given metadata variables
    """
    valid_variables = deepcopy(metadata_variables)

    for name, val in recipe_dict.items():
        if isinstance(val, (int, float)):
            valid_variables[name] = val

        if not isinstance(val, str):
            # only parse string values
            continue

        val = maybe_evaluate_recipe_equation(val, metadata_variables)

        if isinstance(val, (int, float)):
            # update variable value in recipe to be evaluated value, add to valid vars
            recipe_dict[name] = val
            valid_variables[name] = val

    return recipe_dict, valid_variables


def evaluate_recipe_yaml_str_equations(recipe_yaml_str: str) -> str:
    """
    :param recipe_yaml_str: YAML string of a SparseML recipe
    :return: the YAML string with any expressions based on valid
        metadata and recipe variables and operations
    """
    # change references to any classes to become dicts
    pattern = re.compile(r"!(?P<class_name>(?!.*\.)[a-zA-Z_][a-zA-Z^._0-9]+)")
    classless_yaml_str = pattern.sub(r"OBJECT.\g<class_name>:", recipe_yaml_str)

    container = yaml.safe_load(classless_yaml_str)
    if not isinstance(container, dict):
        # yaml string does not create a dict, return original string
        return recipe_yaml_str

    # validate and load metadata
    metadata = container.get("metadata", {})
    metadata, metadata_variables = validate_recipe_metadata(metadata)

    # validate and load remaining variables
    container, variables = evaluate_recipe_variables(container, metadata_variables)

    # update values nested in modifier lists based on the variables
    for key, val in container.items():
        if "modifiers" not in key:
            continue
        container[key] = maybe_evaluate_yaml_object(val, variables)

    updated_yaml_str = yaml.dump(container)

    # convert object dicts back to object declarations and return
    pattern = re.compile(r"OBJECT\.(?P<class_name>(?!.*\.)[a-zA-Z_][a-zA-Z^._0-9]+):")
    return pattern.sub(r"!\g<class_name>", updated_yaml_str)


def _raise_invalid_eval_exception(equation: str):
    raise ValueError(
        f"Unable to evaluation invalid recipe equation value: {equation}. "
        "Valid equations for variables outside of modifiers may only include "
        "metadata variables, numbers, and arithmetic operators. Equations for "
        "modifier params may also include previously defined variable names"
    )


def _maybe_parse_number(val: str) -> Union[str, float, int]:
    try:
        return int(val)
    except Exception:
        try:
            return float(val)
        except Exception:
            return val
