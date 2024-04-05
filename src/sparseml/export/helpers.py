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
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Union

from sparseml.exporters import ExportTargets
from sparsezoo import Model
from sparsezoo.utils.onnx import load_model, onnx_includes_external_data, save_onnx


__all__ = [
    "apply_optimizations",
    "create_deployment_folder",
    "AVAILABLE_DEPLOYMENT_TARGETS",
    "ONNX_MODEL_NAME",
    "create_export_kwargs",
    "save_model_with_external_data",
    "process_source_path",
    "onnx_data_files",
]

AVAILABLE_DEPLOYMENT_TARGETS = [target.value for target in ExportTargets]
ONNX_MODEL_NAME = "model.onnx"
ONNX_DATA_NAME = "model.data"

_LOGGER = logging.getLogger(__name__)


def process_source_path(source_path: Union[Path, str]) -> str:
    """
    Format the source path to be an absolute posix path.
    If the source path is a zoo stub, return the path to
    the training directory
    """
    if isinstance(source_path, str):
        if source_path.startswith("zoo:"):
            source_path = Model(source_path).training.path

            return source_path
        source_path = Path(source_path)
    source_path = source_path.absolute()
    if not source_path.is_dir():
        raise ValueError(
            f"Argument: source_path must be a directory. " f"Got {source_path} instead."
        )
    return source_path.as_posix()


def onnx_data_files(onnx_data_name: str, path: Union[str, Path]) -> List[str]:
    """
    Given the onnx_data_name, return a list of all the onnx data file names
    in the src_path. E.g. if onnx_data_name is "model.data", return
    ["model.data"] (if there is only one file present),
    alternativelty potentially return ["model.data.0", "model.data.1", ...]
    if the files are split into multiple files.

    :param onnx_data_name: The name of the onnx data file.
    :param path: The path to the onnx data file.
    :return: A list of all the onnx data file names.
    """
    onnx_data_pattern = re.compile(rf"{onnx_data_name}(\.\d+)?$")
    onnx_data_files = [
        file for file in os.listdir(path) if onnx_data_pattern.match(file)
    ]
    return onnx_data_files


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
    target_path: Union[Path, str],
    deployment_directory_files_mandatory: List[str],
    source_path: Union[Path, str, None] = None,
    source_config: Optional["PreTrainedConfig"] = None,  # noqa F401
    source_tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa F401
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
    :param source_path: The path to the source folder (where the original model
        files are stored)
    :param source_config: Optional Hugging Face config to copy to deployment dir
    :param source_tokenizer: Optional Hugging Face tokenizer to copy to deployment dir
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

    # prepare for moving the data
    deployment_directory_files_optional = deployment_directory_files_optional or []
    deployment_directory_files_mandatory.remove(ONNX_MODEL_NAME)

    # move the model and (if required) the data files
    move_onnx_files(
        target_path=target_path,
        deployment_folder_dir=deployment_folder_dir,
        onnx_model_name=onnx_model_name,
    )

    if source_path is None:
        # exporting an instantiated model
        if source_config is not None:
            source_config.save_pretrained(deployment_folder_dir)
        if source_tokenizer is not None:
            source_tokenizer.save_pretrained(deployment_folder_dir)
        return deployment_folder_dir

    # exporting from a source path, copy the relevant files to deployment directory
    for file_name in deployment_directory_files_mandatory:
        copy_mandatory_deployment_files(
            file_name, source_path, target_path, onnx_model_name, deployment_folder_dir
        )

    for file_name in deployment_directory_files_optional:
        copy_optional_deployment_files(file_name, source_path, deployment_folder_dir)

    return deployment_folder_dir


def move_onnx_files(
    target_path: Union[str, Path],
    deployment_folder_dir: str,
    onnx_model_name: Optional[str] = None,
):
    onnx_model_name = onnx_model_name or ONNX_MODEL_NAME
    _move_onnx_model(
        onnx_model_name=onnx_model_name,
        src_path=target_path,
        target_path=deployment_folder_dir,
    )


def copy_mandatory_deployment_files(
    file_name: str,
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    onnx_model_name: str,
    deployment_folder_dir: Union[Path, str],
):

    _copy_file_or_directory(
        src=os.path.join(source_path, file_name),
        target=os.path.join(deployment_folder_dir, file_name),
    )


def copy_optional_deployment_files(
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


def save_model_with_external_data(
    onnx_file_path: Union[str, Path], external_data_chunk_size_mb: Optional[int] = None
):
    onnx_model = load_model(onnx_file_path)
    if external_data_chunk_size_mb is not None:
        _LOGGER.debug(
            "Splitting the model into "
            f"{os.path.basename(onnx_file_path)} (graph definition) and one or more "
            f"{ONNX_DATA_NAME} files (constant tensor data). The size of each "
            f"{ONNX_DATA_NAME} file will not exceed {external_data_chunk_size_mb} MB.",
        )
        save_onnx(
            onnx_model,
            onnx_file_path,
            external_data_file=ONNX_DATA_NAME,
            max_external_data_chunk_size=external_data_chunk_size_mb * 1024 * 1024,
        )

    elif onnx_includes_external_data(onnx_model):
        _LOGGER.debug(
            "Splitting the model into"
            f"{os.path.basename(onnx_file_path)} (graph definition) and one or more "
            f"{ONNX_DATA_NAME} files (constant tensor data)"
        )
        save_onnx(onnx_model, onnx_file_path, external_data_file=ONNX_DATA_NAME)

    else:
        _LOGGER.debug(
            "save_with_external_data = True ignored, the model already "
            "has been saved with external data"
        )


def _move_onnx_model(
    onnx_model_name: str, src_path: Union[str, Path], target_path: Union[str, Path]
):
    # move the data file(s)
    for onnx_data_file in onnx_data_files(
        onnx_data_name=onnx_model_name.replace(".onnx", ".data"), path=src_path
    ):
        _move_file(
            src=os.path.join(src_path, onnx_data_file),
            target=os.path.join(target_path, onnx_data_file),
        )
    # move the model itself
    _move_file(
        src=os.path.join(src_path, onnx_model_name),
        target=os.path.join(target_path, onnx_model_name),
    )


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
