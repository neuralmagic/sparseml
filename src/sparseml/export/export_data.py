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
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm


__all__ = ["create_data_samples"]


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
    data_loader: "torch.utils.data.DataLoader",  # noqa F821
    model: Optional["torch.nn.Module"] = None,  # noqa F821
    num_samples: int = 1,
) -> Tuple[
    List["torch.Tensor"],  # noqa F821
    Optional[List["torch.Tensor"]],  # noqa F821
    List["torch.Tensor"],  # noqa F821
]:
    """
    Fetch a batch of samples from the data loader and return the inputs and outputs

    :param data_loader: The data loader to get a batch of inputs/outputs from.
    :param num_samples: The number of samples to generate. Defaults to 1
    :return: The inputs and outputs as lists of torch tensors
    """
    inputs, outputs, labels = [], [], []
    for batch_num, (inputs_, labels_) in tqdm(enumerate(data_loader)):
        if batch_num == num_samples:
            break
        if model:
            outputs_ = model(inputs_)
            outputs.append(outputs_)
        inputs.append(inputs_)
        labels.append(labels_)

    return inputs, outputs, labels


def export_data_samples(
    target_path: Union[Path, str],
    input_samples: Optional[List["torch.Tensor"]] = None,  # noqa F821
    output_samples: Optional[List["torch.Tensor"]] = None,  # noqa F821
    label_samples: Optional[List["torch.Tensor"]] = None,  # noqa F821
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

    for samples, names in zip(
        [input_samples, output_samples, label_samples],
        [InputsNames, OutputsNames, LabelNames],
    ):
        if samples is not None:
            export_data_sample(samples, names, target_path, as_tar)


def export_data_sample(
    samples, names: Enum, target_path: Union[Path, str], as_tar: bool = False
):
    from sparseml.pytorch.utils.helpers import tensors_export, tensors_to_device

    samples = tensors_to_device(samples, "cpu")

    tensors_export(
        tensors=samples,
        export_dir=os.path.join(target_path, names.basename.value),
        name_prefix=names.filename.value,
    )
    if as_tar:
        folder_path = os.path.join(target_path, names.basename.value)
        with tarfile.open(folder_path + ".tar.gz", "w:gz") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))
        shutil.rmtree(folder_path)
