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
import json
import logging
import platform
import re
from contextlib import suppress
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from sparseml import version as sparseml_version
from sparseml.utils import (
    FRAMEWORK_METADATA_KEY,
    RECIPE_METADATA_KEY,
    UnknownVariableException,
    restricted_eval,
)
from sparsezoo import File, Model


__all__ = [
    "load_recipe_yaml_str",
    "load_recipe_yaml_str_no_classes",
    "rewrite_recipe_yaml_string_with_classes",
    "update_recipe_variables",
    "evaluate_recipe_yaml_str_equations",
    "parse_recipe_variables",
    "check_if_staged_recipe",
    "validate_metadata",
    "add_framework_metadata",
]


def load_recipe_yaml_str(
    file_path: str,
    **variable_overrides,
) -> str:
    """
    Loads a YAML recipe file to a string or
    extracts recipe from YAML front matter in a sparsezoo markdown recipe card.
    Recipes can also be provided as SparseZoo model stubs or Recipe
    objects.

    YAML front matter: https://jekyllrb.com/docs/front-matter/

    :param file_path: file path to recipe YAML file or markdown recipe card or
        stub to a SparseZoo model whose recipe will be downloaded and loaded.
        SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
        '?recipe_type=<type>' parameter or include a `/<type>` subpath. Can also
        be a SparseZoo File object. i.e. '/path/to/local/recipe.md',
        'zoo:model/stub/path', 'zoo:model/stub/path?recipe_type=transfer_learn',
        'zoo:model/stub/path/transfer_learn'. Additionally, a raw
         yaml str is also supported in place of a file path.
    :param variable_overrides: dict of variable values to replace
        in the loaded yaml string. Default is None
    :return: the recipe YAML configuration loaded as a string
    """
    if isinstance(file_path, File):
        # download and unwrap Recipe object
        file_path = file_path.path

    if not isinstance(file_path, str):
        raise ValueError(f"file_path must be a str, given {type(file_path)}")

    if file_path.startswith("zoo:"):
        # download from zoo stub
        model = Model(file_path)
        file_path = model.recipes.default.path

    # load the yaml string
    if "\n" in file_path or "\r" in file_path:
        # treat as raw yaml passed in
        yaml_str = file_path
        extension = "unknown"
    else:
        # load yaml from file_path
        extension = file_path.lower().split(".")[-1]
        if extension not in ["md", "yaml"]:
            raise ValueError(
                "Unsupported file extension for recipe. Excepted '.md' or '.yaml'. "
                f"Received {file_path}"
            )
        with open(file_path, "r") as yaml_file:
            yaml_str = yaml_file.read()

    if extension == "md" or extension == "unknown":
        # extract YAML front matter from markdown recipe card
        # adapted from
        # https://github.com/jonbeebe/frontmatter/blob/master/frontmatter
        yaml_delim = r"(?:---|\+\+\+)"
        yaml = r"(.*?)"
        re_pattern = r"^\s*" + yaml_delim + yaml + yaml_delim
        regex = re.compile(re_pattern, re.S | re.M)
        result = regex.search(yaml_str)

        if result:
            yaml_str = result.group(1)
        elif extension == "md":
            # fail if we know whe should have extracted front matter out
            raise RuntimeError(
                "Could not extract YAML front matter from recipe card:"
                " {}".format(file_path)
            )

    if variable_overrides:
        yaml_str = update_recipe_variables(yaml_str, variable_overrides)

    return yaml_str


def load_recipe_yaml_str_no_classes(recipe_yaml_str: str) -> str:
    """
    :param recipe_yaml_str: YAML string of a SparseML recipe
    :return: recipe loaded into YAML with all objects replaced
        as a dictionary of their parameters
    """
    pattern = re.compile(r"!(?P<class_name>(?!.*\.)[a-zA-Z_][a-zA-Z^._0-9]+)")
    classless_yaml_str = pattern.sub(r"OBJECT.\g<class_name>:", recipe_yaml_str)
    return yaml.safe_load(classless_yaml_str)


def rewrite_recipe_yaml_string_with_classes(recipe_contianer: Any) -> str:
    """
    :param recipe_contianer: recipe loaded as yaml with load_recipe_yaml_str_no_classes
    :return: recipe serialized into YAML with original class values re-added
    """
    updated_yaml_str = yaml.dump(recipe_contianer)

    # convert object dicts back to object declarations and return
    pattern = re.compile(
        r"OBJECT\.(?P<class_name>(?!.*\.)[a-zA-Z_][a-zA-Z^._0-9]+):( null)?"
    )
    return pattern.sub(r"!\g<class_name>", updated_yaml_str)


def parse_recipe_variables(
    recipe_variables: Optional[Union[Dict[str, Any], str]] = None
) -> Dict[str, Any]:
    """
    Parse input recipe_variables into a dictionary that can be used to overload
    variables at the root of a recipe.
    Supports dictionaries as well as parsing a string in either json or
    csv key=value format

    :param recipe_variables: the recipe_variables string or dictionary to parse
        for variables used with overloading recipes
    :return: the parsed recipe variables
    """
    if not recipe_variables:
        return {}

    if isinstance(recipe_variables, Dict):
        return recipe_variables

    if not isinstance(recipe_variables, str):
        raise ValueError(
            f"recipe_args must be a string for parsing, given {recipe_variables}"
        )

    # assume json first, try and parse
    with suppress(Exception):
        recipe_variables = json.loads(recipe_variables)
        return recipe_variables

    # assume csv, and standardize to format key=val
    orig_recipe_variables = recipe_variables
    recipe_vars_str = recipe_variables.replace(":", "=")
    recipe_variables = {}
    for arg_val in recipe_vars_str.split(","):
        vals = arg_val.split("=")
        if len(vals) != 2:
            raise ValueError(
                "Improper key=val given in csv for recipe variables with value "
                f"{arg_val} in {orig_recipe_variables}"
            )
        key = vals[0].strip()
        if any(char in key for char in ["{", "!", "=", "}"]):
            raise ValueError(
                "Improper key given in csv for recipe variables with value "
                f"{key} in {orig_recipe_variables}"
            )
        val = vals[1].strip()
        with suppress(Exception):
            # check if val should be a number, otherwise fall back on string
            val = float(val)
        recipe_variables[key] = val

    return recipe_variables


def _update_staged_recipe_variable(var_name, var_value, container):

    # Extends the functionality of _update_container_value to handle staged recipes.

    # :param var_name: The key we are attempting to find in the container.
    # :param var_value: The value which will overwrite the previous value
    #     every time var_name is found in the container's keys.
    # :param container: A container generated from a YAML string of SparseML recipe
    # :return: (optionally mutated) container, as well as key_value
    #     (True if var_key found in container's attributes, otherwise False)

    key_found = False

    for container_key, container_value in container.items():
        if not isinstance(container_value, dict):
            # checking contents of global variables
            if var_name == container_key:
                container[var_name] = var_value
                key_found = True

        else:
            # checking contents of a stage
            stage_container, stage_key_found = _update_recipe_variable(
                var_name, var_value, container_value
            )
            container[container_key] = stage_container
            if stage_key_found:
                key_found = True

    return container, key_found


def _update_recipe_variable(var_name: str, var_value, container):

    # Checks whether there is at least one attribute (key) in the container,
    # with the same name as var_name.
    # If this is the case, key_found = True, otherwise False.
    # Additionally, during checking, when var_name key is present in any of the
    # container's attributes, this attribute's value gets overwritten with var_value.

    # :param var_name: The key we are attempting to find in the container.
    # :param var_value: The value which will overwrite the previous value
    #   every time var_name is found in the container's keys.
    # :param container: A container generated from a YAML string of SparseML recipe
    # :return: (optionally mutated) container, as well as key_value
    #    (True if var_key found in container's attributes, otherwise False)

    key_found = False

    for key, value in container.items():
        if not isinstance(value, list) or not all(
            isinstance(val, dict) for val in value
        ):
            if var_name == key:
                container[var_name] = var_value
                key_found = True
        else:
            for idx, modifier in enumerate(value):
                if var_name in modifier.keys():
                    container[key][idx][var_name] = var_value
                    key_found = True

    return container, key_found


def update_recipe_variables(recipe_yaml_str: str, variables: Dict[str, Any]) -> str:
    """
    :param recipe_yaml_str: YAML string of a SparseML recipe
    :param variables: variables dictionary to update recipe top level variables with.
        If recipe contains stages, it will parse the whole recipe
        and substitute ANY variable with the corresponding name.
    :return: given recipe with variables updated
    """

    container = load_recipe_yaml_str_no_classes(recipe_yaml_str)
    if not isinstance(container, dict):
        # yaml string does not create a dict, return original string
        return recipe_yaml_str

    for var_key, var_value in variables.items():
        if check_if_staged_recipe(container):
            container, key_found = _update_staged_recipe_variable(
                var_key, var_value, container
            )
        else:
            container, key_found = _update_recipe_variable(
                var_key, var_value, container
            )

        if not key_found:
            raise ValueError(
                f"updating recipe variable {var_key} but "
                f"{var_key} is not currently "
                "set in existing recipe. Set the variable in the recipe in order "
                "to overwrite it."
            )

    return rewrite_recipe_yaml_string_with_classes(container)


def evaluate_recipe_yaml_str_equations(recipe_yaml_str: str) -> str:
    """
    :param recipe_yaml_str: YAML string of a SparseML recipe
    :return: the YAML string with any expressions based on valid
        metadata and recipe variables and operations
    """
    container = load_recipe_yaml_str_no_classes(recipe_yaml_str)
    if not isinstance(container, dict):
        # yaml string does not create a dict, return original string
        return recipe_yaml_str

    # check whether the recipe is a stage recipe of not
    if check_if_staged_recipe(container):
        container = _evaluate_staged_recipe_yaml_str_equations(container)

    else:
        container, variables, non_val_variables = _evaluate_container_variables(
            container
        )

        # update values nested in modifier lists based on the variables
        for key, val in container.items():
            if "modifiers" not in key:
                continue
            container[key] = _maybe_evaluate_yaml_object(
                val, variables, non_val_variables
            )

    return rewrite_recipe_yaml_string_with_classes(container)


def check_if_staged_recipe(container: dict) -> bool:
    """
    Check whether container pertains to a staged recipe.
    Such a "staged container" fulfills two conditions:
    - no top level key in container contains "modifiers" in its name
    - a stage should map to a dict that has at least one key with
      "modifiers" in its name
    :param container: a container generated from a YAML string of SparseML recipe
    :return: True if stage recipe, False if normal recipe
    """
    for k, v in container.items():
        if isinstance(v, dict):
            if any(
                key for key in v.keys() if isinstance(key, str) and "modifiers" in key
            ):
                return True
    return False


def _evaluate_staged_recipe_yaml_str_equations(container: dict) -> dict:
    """
    Consumes a staged container and transforms it into a valid
    container for the manager and modifiers to consume further.

    :param container: a staged container generated from a staged recipe.
    :return: transformed container containing evaluated
            variables, operations and objects.
    """
    main_container = {}
    for k, v in container.items():
        if isinstance(v, dict):
            if any([key for key in v.keys() if "modifiers" in key]):
                continue
        main_container.update({k: v})

    stages = {k: container[k] for k in set(container) - set(main_container)}

    (
        main_container,
        global_variables,
        global_non_val_variables,
    ) = _evaluate_container_variables(main_container)

    for stage_name, staged_container in stages.items():
        stage_container, variables, non_val_variables = _evaluate_container_variables(
            staged_container, main_container
        )

        """
        if same variable is both in global_variables and variables, the
        global_variable will get overwritten.
        """
        _global_variables = {
            k: v for k, v in global_variables.items() if k not in variables.keys()
        }
        variables = {**variables, **_global_variables}

        _global_non_val_variables = {
            k: v
            for k, v in global_non_val_variables.items()
            if k not in non_val_variables.keys()
        }
        non_val_variables = {**non_val_variables, **_global_non_val_variables}

        for key, val in staged_container.items():
            if "modifiers" not in key:
                continue
            stage_container[key] = _maybe_evaluate_yaml_object(
                val, variables, non_val_variables
            )

        container[stage_name] = staged_container

    return container


def is_eval_string(val: str) -> bool:
    return val.startswith("eval(") and val.endswith(")")


def _is_evaluatable_variable(val: Any):
    return isinstance(val, (int, float)) or (
        isinstance(val, str) and is_eval_string(val)
    )


def _maybe_evaluate_recipe_equation(
    val: str,
    variables: Dict[str, Union[int, float]],
    non_eval_variables: Dict[str, Any],
    global_container: Optional[Dict[str, Any]] = {},
) -> Union[str, float, int]:
    if is_eval_string(val):
        is_eval_str = True
        val = val[5:-1]
    else:
        return val

    if val in non_eval_variables:
        return non_eval_variables[val]

    if val in global_container:
        return global_container[val]

    evaluated_val = restricted_eval(val, variables)

    if is_eval_str and not isinstance(evaluated_val, (int, float)):
        raise RuntimeError(
            "eval expressions in recipes must evaluate to a float or int"
        )

    return evaluated_val


def _evaluate_container_variables(
    recipe_container: Dict[str, Any], global_container: Optional[Dict[str, Any]] = {}
) -> Tuple[Dict[str, Any], Dict[str, Union[int, float]]]:
    valid_variables = {}
    non_evaluatable_variables = {}
    prev_num_variables = -1

    while prev_num_variables != len(valid_variables):
        prev_num_variables = len(valid_variables)

        for name, val in recipe_container.items():
            if name in valid_variables:
                continue

            if isinstance(val, (int, float)):
                valid_variables[name] = val
                continue

            if not _is_evaluatable_variable(val):
                # only parse string values
                non_evaluatable_variables[name] = val
                continue

            try:
                val = _maybe_evaluate_recipe_equation(
                    val, valid_variables, non_evaluatable_variables, global_container
                )
            except UnknownVariableException:
                # dependant variables maybe not evaluated yet
                continue

            if isinstance(val, (int, float)):
                # update variable value and add to valid vars
                recipe_container[name] = val
                valid_variables[name] = val

    # check that all eval statements have been evaluated
    for name, val in recipe_container.items():
        if isinstance(val, str) and is_eval_string(val):
            raise RuntimeError(
                f"Unable to evaluate expression: {val}. Check if any dependent "
                "variables form a cycle or are not defined"
            )

    return recipe_container, valid_variables, non_evaluatable_variables


def _maybe_evaluate_yaml_object(
    obj: Any,
    variables: Dict[str, Union[int, float]],
    non_eval_variables: Dict[str, Any],
) -> Any:
    if isinstance(obj, str):
        return _maybe_evaluate_recipe_equation(obj, variables, non_eval_variables)
    elif isinstance(obj, list):
        return [
            _maybe_evaluate_yaml_object(val, variables, non_eval_variables)
            for val in obj
        ]
    elif isinstance(obj, dict):
        return {
            key: _maybe_evaluate_yaml_object(val, variables, non_eval_variables)
            for key, val in obj.items()
        }
    else:
        return obj


def _maybe_parse_number(val: str) -> Union[str, float, int]:
    try:
        return int(val)
    except Exception:
        try:
            return float(val)
        except Exception:
            return val


def _extract_metadata_from_recipe(container):
    metadata = {}
    if RECIPE_METADATA_KEY in container.keys():
        metadata = container[RECIPE_METADATA_KEY]
    return metadata


def _extract_metadata_from_staged_recipe(container):
    metadata = {}
    stage_names = _get_recipe_stage_names(container)
    for stage_name in stage_names:
        if RECIPE_METADATA_KEY in container[stage_name].keys():
            metadata[stage_name] = container[stage_name][RECIPE_METADATA_KEY]
    if metadata and (len(metadata) != len(stage_names)):
        raise ValueError(
            "It seems that some stages in your checkpoint recipe"
            "contain metadata and some do not. Either all or no stages must "
            f"must contain the {RECIPE_METADATA_KEY} key"
        )

    return metadata


def _get_recipe_stage_names(container):
    # Extracts valid stage names from a container.
    # Valid stage name (key) is the one which corresponds to a value, which is
    # a dictionary where at least one of the keys contains a string 'modifiers'.

    stage_names = [
        stage_name
        for stage_name, stage_dict in container.items()
        if isinstance(stage_dict, dict)
        and any([key for key in stage_dict.keys() if "modifiers" in key])
    ]
    return stage_names


def _check_warn_dict_difference(original_dict, new_dict):
    if original_dict != new_dict:
        logging.warning(
            f"Attempting to overwrite the previous metadata: {original_dict} "
            f"with new metadata: {new_dict}. "
            "This may lead to different results than the original run of the recipe. "
            "The previous metadata will be omitted and discarded. Ignore if a "
            "change in metadata is expected."
        )
    return new_dict


def add_framework_metadata(
    metadata: Dict[str, Dict], **extra_metadata
) -> Dict[str, Dict]:
    """
    Adds the information (in the form of a nested dictionary)
    about the relevant frameworks used by the user to the metadata.
    :param metadata: Validated metadata
    :param extra_metadata: Optional framework metadata, specific for the given framework
        (e.g. for pytorch integration
        'add_framework_metadata(metadata, pytorch_version = torch.__version__)')
    :return: Validated metadata with framework metadata
    """

    framework_metadata = {
        "python_version": platform.python_version(),
        "sparseml_version": sparseml_version,
    }

    framework_metadata.update(extra_metadata)

    for stage_name, stage_value in metadata.items():
        if stage_value is None:
            stage_metadata = {FRAMEWORK_METADATA_KEY: framework_metadata}
        else:
            if (FRAMEWORK_METADATA_KEY in stage_value.keys()) and stage_value[
                FRAMEWORK_METADATA_KEY
            ]:
                shared_keys = set(
                    stage_value[FRAMEWORK_METADATA_KEY].keys()
                ).intersection(set(framework_metadata.keys()))
                warning_if_stage = (
                    f"stage (stage name: {stage_name})"
                    if stage_name != RECIPE_METADATA_KEY
                    else ""
                )
                warning_msg = (
                    f"Overwriting metadata {warning_if_stage} key(s) "
                    f"{shared_keys} with new value(s) "
                    f"{ {k:v for k,v in framework_metadata.items() if k in shared_keys} }"  # noqa E501
                )
                logging.warning(warning_msg)
            stage_metadata = deepcopy(stage_value)
            stage_metadata[FRAMEWORK_METADATA_KEY] = framework_metadata
        metadata[stage_name] = stage_metadata

    return metadata


def validate_metadata(metadata: dict, yaml_str: str) -> dict:
    """
    Compare the metadata (previous_metadata) carried over from the recipe
    (`yaml_str`) with the new, incoming metadata ('metadata').

    If attempting to overwrite previous metadata with the new metadata,
    the script throws a warning and overwrites the previous metadata.
    Otherwise, it propagates the new metadata in the correct form.

    :param metadata: New metadata
    :param yaml_str: String representation of the recipe YAML file,
        (may contain previous metadata)
    :return: Validated metadata
    """

    container = load_recipe_yaml_str_no_classes(yaml_str)
    is_container_staged = check_if_staged_recipe(container)

    checkpoint_metadata = (
        _extract_metadata_from_staged_recipe(container)
        if is_container_staged
        else _extract_metadata_from_recipe(container)
    )

    if checkpoint_metadata:
        if metadata:
            if is_container_staged:

                is_metadata_staged = set(_get_recipe_stage_names(container)) == set(
                    metadata.keys()
                )

                for stage_name in _get_recipe_stage_names(container):
                    if is_metadata_staged:
                        checkpoint_metadata[stage_name] = _check_warn_dict_difference(
                            container[stage_name][RECIPE_METADATA_KEY],
                            metadata[stage_name],
                        )
                    else:
                        checkpoint_metadata[stage_name] = _check_warn_dict_difference(
                            container[stage_name][RECIPE_METADATA_KEY], metadata
                        )

            else:
                checkpoint_metadata = _check_warn_dict_difference(
                    container[RECIPE_METADATA_KEY], metadata
                )

        return (
            checkpoint_metadata
            if is_container_staged
            else {RECIPE_METADATA_KEY: checkpoint_metadata}
        )

    else:
        if metadata:

            return (
                {
                    stage_name: metadata
                    for stage_name in _get_recipe_stage_names(container)
                }
                if is_container_staged
                else {RECIPE_METADATA_KEY: metadata}
            )

        else:
            return (
                {stage_name: None for stage_name in _get_recipe_stage_names(container)}
                if is_container_staged
                else {RECIPE_METADATA_KEY: None}
            )
