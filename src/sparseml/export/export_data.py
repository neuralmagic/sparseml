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
from typing import Any, List, Optional, Tuple, Union

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


def create_data_samples(
    data_loader: torch.utils.data.DataLoader,
    model: Optional[torch.nn.Module] = None,
    num_samples: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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
        if len(data) == 2:
            inputs_, labels_ = data
        else:
            inputs_ = {key: value.to("cpu") for key, value in data.items()}
            labels_ = None
        if batch_num == num_samples:
            break
        if model:
            if labels_ is not None:
                outputs_ = model(inputs_)
            else:
                outputs_ = model(**inputs_).end_logits

            if isinstance(outputs_, tuple):
                # outputs_ contains (logits, softmax)
                outputs_ = outputs_[0]
            outputs.append(outputs_)
        inputs.append(inputs_)
        if labels_ is not None:
            labels.append(
                torch.IntTensor([labels_])
                if not isinstance(labels_, torch.Tensor)
                else labels_
            )

    return inputs, outputs, labels


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
