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

import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Union

from sparseml.exporters import ExportTargets


__all__ = [
    "apply_optimizations",
    "create_deployment_folder",
    "AVAILABLE_DEPLOYMENT_TARGETS",
    "ONNX_MODEL_NAME",
    "create_export_kwargs",
    "format_source_path",
]

AVAILABLE_DEPLOYMENT_TARGETS = [target.value for target in ExportTargets]
ONNX_MODEL_NAME = "model.onnx"
ONNX_DATA_NAME = "model.data"

_LOGGER = logging.getLogger(__name__)


def format_source_path(source_path: Union[Path, str]) -> str:
    """
    Format the source path to be an absolute posix path
    """
    if isinstance(source_path, str):
        source_path = Path(source_path)
    source_path = source_path.absolute()
    if not source_path.is_dir():
        raise ValueError(
            f"Argument: source_path must be a directory. " f"Got {source_path} instead."
        )
    return source_path.as_posix()


def create_export_kwargs(
    loaded_model_kwargs: Dict[str, Any], export_target: str = "deepsparse"
) -> Dict[str, Any]:
    """
    Retrieve the export kwargs from the loaded model kwargs.

    The export kwargs are the kwargs that are passed to the export function.
    Given the loaded model kwargs and the export_target, one can define which
    loaded_model_kwargs should be routed to the export kwargs.

    :param loaded_model_kwargs: The loaded model kwargs.
    :param export_target: The export target.
    :return: The export kwargs.
    """

    if export_target not in AVAILABLE_DEPLOYMENT_TARGETS:
        raise ValueError(
            f"Export target {export_target} not in "
            f"available targets {AVAILABLE_DEPLOYMENT_TARGETS}"
        )

    export_kwargs = {}
    input_names = loaded_model_kwargs.get("input_names")
    if input_names is not None:
        export_kwargs["input_names"] = input_names

    return export_kwargs


def create_deployment_folder(
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    deployment_directory_files_mandatory: List[str],
    deployment_directory_files_optional: Optional[List[str]] = None,
    deployment_directory_name: str = "deployment",
    onnx_model_name: Optional[str] = None,
) -> str:
    """
    Copy the relevant files to the deployment folder.

    The deployment folder will be created at target_path/deployment_directory_name.
    The relevant files are copied from:
    - if file is an ONNX model (or ONNX data file), the file will be copied
        from target_path
    - else, the file will be copied from source_path

    :param source_path: The path to the source folder. This is where the ONNX model
        and (optionally) ONNX data file are located.
    :param target_path: The path to the target folder.
    :param deployment_directory_name: The name of the deployment directory.
        The files will be copied to target_path/deployment_directory_name.
    :param deployment_directory_files_mandatory: The mandatory list of files
        to copy to the deployment directory. If the file is an ONNX model
        (or ONNX data file), the file will be copied from target_path.
        Else, the file will be copied from source_path.
    :param deployment_directory_files_optional: The optional list of files
        to copy to the deployment directory.
    :param onnx_model_name: The name of the ONNX model file. If not specified,
        defaults to ONNX_MODEL_NAME.
    :return: The path to the deployment folder.
    """
    # create the deployment folder
    deployment_folder_dir = os.path.join(target_path, deployment_directory_name)
    if os.path.isdir(deployment_folder_dir):
        shutil.rmtree(deployment_folder_dir)
    os.makedirs(deployment_folder_dir, exist_ok=True)

    deployment_directory_files_optional = deployment_directory_files_optional or []

    for file_name in deployment_directory_files_mandatory:
        move_mandatory_deployment_files(
            file_name, source_path, target_path, onnx_model_name, deployment_folder_dir
        )

    for file_name in deployment_directory_files_optional:
        move_optional_deployment_files(file_name, source_path, deployment_folder_dir)

    return deployment_folder_dir


def move_mandatory_deployment_files(
    file_name: str,
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    onnx_model_name: str,
    deployment_folder_dir: Union[Path, str],
):
    if file_name == ONNX_MODEL_NAME:
        # attempting to move the ONNX model file
        # (potentially together with the ONNX data file)
        # from target_path to target_path/deployment_folder_dir

        # takes into consideration potentially custom ONNX model name
        onnx_model_name = (
            ONNX_MODEL_NAME if onnx_model_name is None else onnx_model_name
        )

        _move_onnx_model(
            onnx_model_name=onnx_model_name,
            src_path=target_path,
            target_path=deployment_folder_dir,
        )

    else:
        _copy_file_or_directory(
            src=os.path.join(source_path, file_name),
            target=os.path.join(deployment_folder_dir, file_name),
        )


def move_optional_deployment_files(
    file_name: str,
    source_path: Union[Path, str],
    deployment_folder_dir: Union[Path, str],
):
    if os.path.exists(os.path.join(source_path, file_name)):
        _copy_file_or_directory(
            src=os.path.join(source_path, file_name),
            target=os.path.join(deployment_folder_dir, file_name),
        )
    else:
        _LOGGER.warning(
            f"Optional file {file_name} not found in source path {source_path}"
        )


class GraphOptimizationOptions(Enum):
    """
    Holds the string names of the graph optimization options.
    """

    none = "none"
    all = "all"


def apply_optimizations(
    onnx_file_path: Union[str, Path],
    available_optimizations: OrderedDict[str, Callable],
    target_optimizations: Union[str, List[str]] = GraphOptimizationOptions.all.value,
    single_graph_file: bool = True,
):
    """
    Apply optimizations to the graph of the ONNX model.

    :param onnx_file_path: The path to the ONNX model file.
    :param available_optimizations: The graph optimizations available
        for the model. It is an ordered mapping from the string names
        to functions that alter the model
    :param target_optimizations: The name(s) of optimizations to apply.
        It can be either a list of string name or a single string option
        that specifies the set of optimizations to apply.
        If is string, refer to the `GraphOptimizationOptions` enum
        for the available options.
    :param single_graph_file: Whether to save the optimized graph to a single
        file or split it into multiple files. By default, it is True.
    """
    optimizations: Dict[str, Callable] = resolve_graph_optimizations(
        available_optimizations=available_optimizations,
        optimizations=target_optimizations,
    )

    for name, optimization in optimizations.items():
        _LOGGER.info(f"Attempting to apply optimization: {name}... ")
        applied = optimization(onnx_file_path)
        if applied:
            _LOGGER.info(
                f"Optimization: {name} has been successfully "
                f"applied to the ONNX model: {onnx_file_path}"
            )

    if not single_graph_file:
        save_onnx_multiple_files(onnx_file_path)


def resolve_graph_optimizations(
    optimizations: Union[str, List[str]],
    available_optimizations: Optional[OrderedDict[str, Callable]] = None,
) -> Dict[str, Callable]:
    """
    Get the optimization functions to apply to the onnx model.

    :param available_optimizations: The graph optimizations available
        for the model. It is an ordered mapping from the string names
        to functions that alter the model
    :param optimizations: The name(s) of optimizations to apply.
        It can be either a list of string name or a single string option
        that specifies the set of optimizations to apply.
        If is string, refer to the `GraphOptimizationOptions` enum
        for the available options.
    :return: The optimization functions to apply to the onnx model.
    """
    if isinstance(optimizations, str):
        if optimizations == GraphOptimizationOptions.none.value:
            return {}
        elif optimizations == GraphOptimizationOptions.all.value:
            return available_optimizations or {}
        else:
            raise KeyError(f"Unknown graph optimization option: {optimizations}")
    elif isinstance(optimizations, list):
        return {name: available_optimizations[name] for name in optimizations}
    else:
        raise KeyError(f"Unknown graph optimization option: {optimizations}")


# TODO: To discuss with @bfineran
def save_onnx_multiple_files(*args, **kwargs):
    raise NotImplementedError


def _move_onnx_model(
    onnx_model_name: str, src_path: Union[str, Path], target_path: Union[str, Path]
):
    onnx_data_name = onnx_model_name.replace(".onnx", ".data")

    onnx_model_path = os.path.join(src_path, onnx_model_name)
    onnx_data_path = os.path.join(src_path, onnx_data_name)

    if os.path.exists(onnx_data_path):
        _move_file(src=onnx_data_path, target=os.path.join(target_path, onnx_data_name))
    _move_file(src=onnx_model_path, target=os.path.join(target_path, onnx_model_name))


def _copy_file_or_directory(src: str, target: str):
    if not os.path.exists(src):
        raise ValueError(
            f"Attempting to copy file from {src}, but the file does not exist."
        )
    if os.path.isdir(src):
        shutil.copytree(src, target)
    else:
        shutil.copyfile(src, target)


def _move_file(src: str, target: str):
    if not os.path.exists(src):
        raise ValueError(
            f"Attempting to move file from {src}, but the file does not exist."
        )
    shutil.move(src, target)
