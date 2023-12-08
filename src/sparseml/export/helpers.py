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
from typing import Callable, List, Union

import onnx

from sparsezoo.utils.onnx import save_onnx


__all__ = ["apply_optimizations", "export_sample_inputs_outputs"]


class GraphOptimizationOptions(Enum):
    """
    Holds the string names of the graph optimization options.
    """

    none = "none"
    all = "all"


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
