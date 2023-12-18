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
import tarfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from sparseml.pytorch.utils.helpers import tensors_export, tensors_to_device


__all__ = ["create_data_samples", "export_data_samples"]

_LOGGER = logging.getLogger(__name__)


class LabelNames(Enum):
    basename = "sample-labels"
    filename = "lab"


class OutputsNames(Enum):
    basename = "sample-outputs"
    filename = "out"


class InputsNames(Enum):
    basename = "sample-inputs"
    filename = "inp"


def export_data_samples(
    target_path: Union[Path, str],
    input_samples: Optional[List[Any]] = None,
    output_samples: Optional[List[Any]] = None,
    label_samples: Optional[List[Any]] = None,
    as_tar: bool = False,
):
    """
    Save the input, labels and output samples to the target path.
    All the input files are optional. If a sample is None,
    it will not be saved.

    Input samples will be saved to:
    .../sample-inputs/inp_0001.npz
    .../sample-inputs/inp_0002.npz
    ...

    Output samples will be saved to:
    .../sample-outputs/out_0001.npz
    .../sample-outputs/out_0002.npz
    ...

    Label samples will be saved to:
    .../sample-labels/lab_0001.npz
    .../sample-labels/lab_0002.npz
    ...

    If as_tar is True, the samples will be saved as tar files:
    .../sample-inputs.tar.gz
    .../sample-outputs.tar.gz
    .../sample-labels.tar.gz

    :param input_samples: The input samples to save.
    :param output_samples: The output samples to save.
    :param label_samples: The label samples to save.
    :param target_path: The path to save the samples to.
    :param as_tar: Whether to save the samples as tar files.
    """

    for samples, names in zip(
        [input_samples, output_samples, label_samples],
        [InputsNames, OutputsNames, LabelNames],
    ):
        if len(samples) > 0:
            _LOGGER.info(f"Exporting {names.basename.value} to {target_path}...")
            break_batch = isinstance(samples[0], dict)
            export_data_sample(samples, names, target_path, as_tar, break_batch)
            _LOGGER.info(
                f"Successfully exported {names.basename.value} to {target_path}!"
            )


def export_data_sample(
    samples,
    names: Enum,
    target_path: Union[Path, str],
    as_tar: bool = False,
    break_batch=False,
):

    samples = tensors_to_device(samples, "cpu")

    tensors_export(
        tensors=samples,
        export_dir=os.path.join(target_path, names.basename.value),
        name_prefix=names.filename.value,
        break_batch=break_batch,
    )
    if as_tar:
        folder_path = os.path.join(target_path, names.basename.value)
        with tarfile.open(folder_path + ".tar.gz", "w:gz") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))
        shutil.rmtree(folder_path)


def create_data_samples(
    data_loader: torch.utils.data.DataLoader,
    model: Optional[torch.nn.Module] = None,
    num_samples: int = 1,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Fetch a batch of samples from the data loader and return the inputs and outputs

    :param data_loader: The data loader to get a batch of inputs/outputs from.
    :param model: The model to run the inputs through to get the outputs.
        If None, the outputs will be an empty list.
    :param num_samples: The number of samples to generate. Defaults to 1
    :return: The inputs and outputs as lists of torch tensors
    """
    inputs, outputs, labels = [], [], []
    if model is None:
        _LOGGER.warning("The model is None. The list of outputs will be empty")

    for batch_num, data in tqdm(enumerate(data_loader)):
        if batch_num == num_samples:
            break
        if isinstance(data, dict):
            inputs_, labels_, outputs_ = run_inference_with_dict_data(
                data=data, model=model
            )
        elif isinstance(data, tuple):
            inputs_, labels_, outputs_ = run_inference_with_tuple_data(
                data=data, model=model
            )

        inputs.append(inputs_)
        if outputs_ is not None:
            outputs.append(outputs_)
        if labels_ is not None:
            labels.append(
                torch.IntTensor([labels_])
                if not isinstance(labels_, torch.Tensor)
                else labels_
            )

    return inputs, outputs, labels


def run_inference_with_dict_data(
    data: Dict[str, Any], model: Optional[torch.nn.Module] = None
) -> Tuple[Dict[str, Any], Any, Optional[Dict[str, Any]]]:
    """
    Run inference on a model by inferring the appropriate
    inputs from the dictionary input data.


    :param data: The data to run inference on
    :param model: The model to run inference on (optional)
    :return: The inputs, labels and outputs
    """
    # TODO: For now we need to make sure that the model and tensors
    # live on the same device. This is because I am currently unable
    # to assign them to the same device (transformer scenario)
    inputs = {key: value.to("cpu") for key, value in data.items()}

    label = None
    if model is None:
        return inputs, label, None

    model.to("cpu")
    output_vals = model(**inputs)
    output = {
        name: torch.squeeze(val).detach().to("cpu") for name, val in output_vals.items()
    }
    return inputs, label, output


def run_inference_with_tuple_data(
    data: Tuple[Any, Any], model: Optional[torch.nn.Module] = None
) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Run inference on a model by inferring the appropriate
    inputs from the tuple input data.

    :param inputs: The data to run inference on
    :param model: The model to run inference on (optional)
    :return: The inputs, labels and outputs
    """
    # assume that
    inputs, labels = data
    outputs = model(inputs) if model else None
    if isinstance(outputs, tuple):
        # outputs_ contains (logits, softmax)
        outputs = outputs[0]
    return inputs, labels, outputs
