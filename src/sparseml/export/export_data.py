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
from collections import OrderedDict
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
            _check_if_samples_already_exist(
                os.path.join(target_path, names.basename.value)
            )
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
        elif isinstance(data, (list, tuple)):
            inputs_, labels_, outputs_ = run_inference_with_tuple_or_list_data(
                data=data, model=model
            )
        else:
            raise ValueError(
                f"Data type {type(data)} is not supported. "
                f"Only dict and tuple are supported"
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

    # turn all the returned lists into a list of dicts
    # to facilitate the sample export
    if inputs and not isinstance(inputs[0], dict):
        inputs = [dict(input=input) for input in inputs]

    if labels and not isinstance(labels[0], dict):
        labels = [dict(label=label) for label in labels]

    if outputs and not isinstance(outputs[0], dict):
        outputs = [dict(output=output) for output in outputs]

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
    labels = None
    if model is None:
        output = None
    else:
        # move the inputs to the model device and
        # grab only the first sample from the batch
        inputs = {
            key: value[0].to(model.device).reshape(1, -1) for key, value in data.items()
        }
        output_vals = model(**inputs)
        output = {
            name: torch.squeeze(val).detach().to("cpu")
            for name, val in output_vals.items()
        }
    inputs = {key: value.to("cpu")[0] for key, value in data.items()}
    return inputs, labels, output


# this function is specific for image-classification for now
# to be generalized later
def run_inference_with_tuple_or_list_data(
    data: Tuple[Any, Any], model: Optional[torch.nn.Module] = None
) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Run inference on a model by inferring the appropriate
    inputs from the tuple input data.

    :param data: The data to run inference on
    :param model: The model to run inference on (optional)
    :return: The inputs, labels and outputs
    """
    inputs, labels = data

    outputs = model(inputs) if model else None
    if isinstance(outputs, tuple):
        # outputs_ contains (logits, scores)
        outputs = OrderedDict(logits=outputs[0], scores=outputs[1])
    if len(inputs.size()) == 4:
        # if the input is a batch, remove the batch dimension
        inputs = torch.squeeze(inputs, 0)
    return inputs, labels, outputs


def _check_if_samples_already_exist(sample_path: Union[str, Path]) -> bool:
    samples_exist = os.path.isdir(sample_path)
    if samples_exist:
        _LOGGER.warning(f"Samples already exist in {sample_path}. Overwriting...")
    return samples_exist
