import json
import os
from typing import Any, Dict, List

import numpy as np
import yaml
from flask import current_app, request

CONFIG_YAML_PATH = "server.yaml"
ALLOWED_EXTENSIONS = {"onnx"}

__all__ = [
    "allowed_file",
    "get_config_yml",
    "get_path",
    "get_missing_fields_message",
    "is_prunable",
    "get_array_from_data",
    "are_array_equal",
    "get_project_root",
]


def get_config_yml() -> Dict:
    with open(CONFIG_YAML_PATH) as yml_file:
        config_yml = yaml.safe_load(yml_file)
    return config_yml


def get_path(filename: str, project_root: str = None) -> str:
    try:
        root = current_app.config["PROJECT_ROOT"]
    except:
        root = get_project_root(project_root)
    current_path = os.path.join(root, filename)
    return os.path.abspath(current_path)


def get_missing_fields_message(required_fields) -> Dict[str, str]:
    for required_field in required_fields:
        if required_field not in request.get_json():
            return {"message": f"Missing required field '{required_field}'"}
    return None


def is_prunable(name: str) -> bool:
    name = name.lower()
    return ("conv" in name) or ("gemm" in name) or "winograd_fused" in name


def get_array_from_data(data: dict) -> List[int]:
    return [data["x"], data["y"], data["z"]]


def are_array_equal(array_one: List, array_two: List) -> bool:
    return (
        (array_one == array_two)
        or (
            len(array_one) > len(array_two)
            and array_one[: len(array_two)] == array_two
            and np.prod(array_one[len(array_two) :]) == 1
        )
        or (
            len(array_two) > len(array_one)
            and array_two[: len(array_one)] == array_one
            and np.prod(array_two[len(array_one) :]) == 1
        )
    )


def get_project_root(arg_project_root: str):
    config_yml = get_config_yml()

    project_root = (
        config_yml["projects-folder"]
        if config_yml and "projects-folder" in config_yml and not arg_project_root
        else arg_project_root
    )

    if project_root is None:
        project_root = "~/nm-projects"

    project_root = os.path.expanduser(project_root)

    if not os.path.exists(project_root):
        raise NotADirectoryError(f"PROJECT_ROOT {project_root} does not exist")

    project_root = os.path.abspath(project_root)
    return project_root


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
