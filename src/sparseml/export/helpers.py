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

import os
import shutil
import tarfile
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import onnx

from sparseml.exporters import ExportTargets
from sparsezoo.utils.onnx import save_onnx


__all__ = [
    "apply_optimizations",
    "export_sample_inputs_outputs",
    "create_deployment_folder",
    "AVAILABLE_DEPLOYMENT_TARGETS",
    "ONNX_MODEL_NAME",
]

AVAILABLE_DEPLOYMENT_TARGETS = [target.value for target in ExportTargets]
ONNX_MODEL_NAME = "model.onnx"
ONNX_DATA_NAME = "model.data"
DEFAULT_DEPLOYMENT_DIRECTORY_STRUCTURE = {ONNX_MODEL_NAME: {}}


def create_deployment_folder(
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    deployment_directory_name: str = "deployment",
    deployment_directory_structure: Optional[Dict] = None,
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
    :param deployment_directory_structure: The structure of the deployment directory.
        If not specified, defaults to DEFAULT_DEPLOYMENT_DIRECTORY_STRUCTURE.
    :param onnx_model_name: The name of the ONNX model file. If not specified,
        defaults to ONNX_MODEL_NAME.
    """

    # create the deployment folder
    deployment_folder_dir = os.path.join(target_path, deployment_directory_name)
    if os.path.isdir(deployment_folder_dir):
        shutil.rmtree(deployment_folder_dir)
    os.makedirs(deployment_folder_dir, exist_ok=True)

    # copy over the expected files
    deployment_directory_structure = (
        deployment_directory_structure or DEFAULT_DEPLOYMENT_DIRECTORY_STRUCTURE
    )

    for name, child_name in deployment_directory_structure.items():
        if child_name:
            # name is not a file name but a directory
            # name that contains files with names
            # specified in child_name
            raise NotImplementedError(
                "Nested deployment directory structure is not supported yet"
            )
        else:
            if name == ONNX_MODEL_NAME:
                # attempting to move the ONNX model file
                # (potentially together with the ONNX data file)
                # from source_path to deployment_folder_dir

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
                _copy_file(
                    src=os.path.join(source_path, name),
                    target=os.path.join(deployment_folder_dir, name),
                )
    return deployment_folder_dir


class OutputsNames(Enum):
    basename = "sample-outputs"
    filename = "out"


class InputsNames(Enum):
    basename = "sample-inputs"
    filename = "inp"


def export_sample_inputs_outputs(
    input_samples: List["torch.Tensor"],  # noqa F821
    output_samples: List["torch.Tensor"],  # noqa F821
    target_path: Union[Path, str],
    as_tar: bool = False,
):
    """
    Save the input and output samples to the target path.

    Input samples will be saved to:
    .../sample-inputs/inp_0001.npz
    .../sample-inputs/inp_0002.npz
    ...

    Output samples will be saved to:
    .../sample-outputs/out_0001.npz
    .../sample-outputs/out_0002.npz
    ...

    If as_tar is True, the samples will be saved as tar files:
    .../sample-inputs.tar.gz
    .../sample-outputs.tar.gz

    :param input_samples: The input samples to save.
    :param output_samples: The output samples to save.
    :param target_path: The path to save the samples to.
    :param as_tar: Whether to save the samples as tar files.
    """

    from sparseml.pytorch.utils.helpers import tensors_export, tensors_to_device

    input_samples = tensors_to_device(input_samples, "cpu")
    output_samples = tensors_to_device(output_samples, "cpu")

    for tensors, names in zip(
        [input_samples, output_samples], [InputsNames, OutputsNames]
    ):
        tensors_export(
            tensors=tensors,
            export_dir=os.path.join(target_path, names.basename.value),
            name_prefix=names.filename.value,
        )
    if as_tar:
        for folder_name_to_tar in [
            InputsNames.basename.value,
            OutputsNames.basename.value,
        ]:
            folder_path = os.path.join(target_path, folder_name_to_tar)
            with tarfile.open(folder_path + ".tar.gz", "w:gz") as tar:
                tar.add(folder_path, arcname=os.path.basename(folder_path))
            shutil.rmtree(folder_path)


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
    optimizations: List[Callable] = resolve_graph_optimizations(
        available_optimizations=available_optimizations,
        optimizations=target_optimizations,
    )

    onnx_model = onnx.load(onnx_file_path)

    for optimization in optimizations:
        onnx_model = optimization(onnx_model)

    if single_graph_file:
        save_onnx(onnx_model, onnx_file_path)
        return

    save_onnx_multiple_files(onnx_model)


def resolve_graph_optimizations(
    available_optimizations: OrderedDict[str, Callable],
    optimizations: Union[str, List[str]],
) -> List[Callable]:
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
    return The list of optimization functions to apply.
    """
    if isinstance(optimizations, str):
        if optimizations == GraphOptimizationOptions.none.value:
            return []
        elif optimizations == GraphOptimizationOptions.all.value:
            return list(available_optimizations.values())
        else:
            raise KeyError(f"Unknown graph optimization option: {optimizations}")
    elif isinstance(optimizations, list):
        return [available_optimizations[optimization] for optimization in optimizations]
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


def _copy_file(src: str, target: str):
    if not os.path.exists(src):
        raise ValueError(
            f"Attempting to copy file from {src}, but the file does not exist."
        )
    shutil.copyfile(src, target)


def _move_file(src: str, target: str):
    if not os.path.exists(src):
        raise ValueError(
            f"Attempting to move file from {src}, but the file does not exist."
        )
    shutil.move(src, target)
