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

import copy
import logging
import shutil
from abc import ABC
from pathlib import Path
from typing import Callable, Dict, Iterable, Union

import torch

from safetensors.torch import save_file
from sparseml.pytorch.model_load.helpers import load_safetensors_state_dict
from sparseml.utils.pytorch.converters.transformations import (
    transform_autogptq_weights_and_reshape_tensors,
    transform_exllama_names,
)


StateDictType = Union[Dict[str, torch.Tensor], str, Path]
TransformationType = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
_LOGGER: logging.Logger = logging.getLogger(__name__)


class BaseConverter(ABC):
    @classmethod
    def translate(cls, state_dict: StateDictType, **kwargs) -> StateDictType:
        """
        Applies transformations to the state_dict

        :param state_dict: The state_dict to apply transformations to
        :param kwargs: Additional arguments to pass to the transformations
        :return: The transformed state_dict
        """
        _LOGGER.info("Applying transformations...")
        new_state_dict = copy.copy(state_dict)
        for transformation in cls.transformations():
            new_state_dict = transformation(new_state_dict, **kwargs)
        return new_state_dict

    @classmethod
    def convert_from_safetensors(cls, filepath: str, save_dir: str = None) -> str:
        """
        Convert a .safetensors file or directory of .safetensors files, applying
        transformations to the state_dict and saving the new state_dict to a new
        directory

        :param filepath: The file path to the .safetensors file or directory
            containing .safetensors files to convert
        :param save_dir: The directory to save the converted state_dict to
        :return: The directory where the converted state_dict was saved
        """
        _validate_safetensors_file_path(filepath)

        filepath_: Path = Path(filepath)
        if not save_dir:
            save_dir = "compressed_tensors_model"

        save_dir_: Path = Path(save_dir)
        save_dir_.mkdir(exist_ok=True, parents=True)

        metadata = {"format": "pt", "source": "Created by SparseML"}

        # transform and save the state_dict
        if filepath_.is_dir():
            for file in filepath_.glob("*.safetensors"):
                _LOGGER.info(f"Loading file: {file}")
                state_dict: StateDictType = load_safetensors_state_dict(file)
                new_state_dict = cls.translate(state_dict=state_dict)
                save_file(
                    new_state_dict, filename=save_dir_ / file.name, metadata=metadata
                )
            _copy_non_safetensor_files_(filepath_, save_dir_)
            _update_quantization_config(filepath_, save_dir_)

        elif filepath_.is_file():
            state_dict: StateDictType = load_safetensors_state_dict(filepath)
            new_state_dict = cls.translate(state_dict=state_dict)
            save_file(
                new_state_dict, save_path=save_dir_ / filepath_.name, metadata=metadata
            )

        return str(save_dir_)

    @classmethod
    def transformations(cls) -> Iterable[TransformationType]:
        """
        Returns an iterable of transformations that are applied in the converter,
        each transformation should be a callable that takes a state_dict and returns
        a transformed state_dict
        """
        raise NotImplementedError()


class ExllamaToCompressedTensorConverter(BaseConverter):
    """
    A converter that applies transformations to the state_dict of a autogptq
    quantized model to convert it to a compressed tensor model, which can be
    loaded by the SparseAutoModel classes
    """

    @classmethod
    def transformations(cls):
        return (transform_autogptq_weights_and_reshape_tensors, transform_exllama_names)


def _validate_safetensors_file_path(filepath: str):
    """
    Given a file path, it is valid if:
        - The file exists
        - The file is either a single .safetensors file or a
            directory containing .safetensors files

    :param filepath: A string file path to validate
    """

    filepath_: Path = Path(filepath)

    if not filepath_.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath_.is_dir() and not any(filepath_.glob("*.safetensors")):
        raise FileNotFoundError(f"No .safetensors files found in directory: {filepath}")

    if filepath_.is_file() and not filepath_.suffix == ".safetensors":
        raise ValueError(f"File must be a .safetensors file: {filepath}")


def _copy_non_safetensor_files_(source_dir: Path, dest_dir: Path):
    """
    A helper function to copy all auxillary files in a directory that are
    not .safetensors files, for example (config.json, recipe.yaml, ...)

    :param source_dir: The directory to copy files from
    :param dest_dir: The directory to copy files to
    """
    for file in source_dir.glob("*"):
        if file.suffix != ".safetensors":
            _LOGGER.info(f"Copying file: {file} to {dest_dir}")
            shutil.copy(file, dest_dir / file.name)


def _update_quantization_config(source_dir: Path, dest_dir: Path):
    """
    Updates config.json file in the destination directory by removing the
    quantization_config attribute

    :param source_dir: The directory containing the original config.json file
    :param dest_dir: The directory to save the updated config.json file
    """
    from sparseml.transformers import SparseAutoConfig

    config = SparseAutoConfig.from_pretrained(source_dir)

    if hasattr(config, "quantization_config"):
        _LOGGER.info("Updating quantization config...")
        delattr(config, "quantization_config")
    config.save_pretrained(dest_dir)
