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

"""
General utilities for exporting models to different formats using safe tensors.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sparseml.transformers.utils.sparse_model import SparseAutoModelForCausalLM
from sparseml.transformers.utils.sparse_tokenizer import SparseAutoTokenizer
from sparseml.transformers.utils.transformations import (
    add_tensors,
    convert_fp32_to_fp16,
    remove_unwanted_tensors,
    transform_names,
    transform_tensors,
)
from sparseml.utils import get_unique_dir_name


__all__ = [
    "export_vllm_checkpoint",
    "SUPPORTED_FORMAT_TYPES",
]

SUPPORTED_FORMAT_TYPES = Literal["exllama", "marlin"]
_LOGGER = logging.getLogger(__name__)


def export_vllm_checkpoint(
    model: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizerBase, str, None] = None,
    format: SUPPORTED_FORMAT_TYPES = "exllama",
    save_dir: Union[str, Path, None] = None,
):
    """
    A utility function to export a GPTQ quantized model to safetensors,
    compatible with the vLLM library.
    Calls the appropriate state dict translation function based on the format
    and saves the translated state dict to the specified directory.
    If the directory is not specified defaults to cwd/exported_model.
    If the directory already exists, a new directory is created with a unique name.

    :param model: The loaded model to be exported, can also be a local model
        directory path, or a HugginFace/SparseZoo stub
    :param tokenizer: The tokenizer associated with the model, can also
        be a HuggingFace/Sparsezoo stub.
    :param format: The format to which the model should be exported.
        Default is "exllama".
    :param save_dir: The directory where the model should be saved.
    """
    # validate
    if format not in SUPPORTED_FORMAT_TYPES:
        raise ValueError(
            f"Unsupported format {format}, supported formats "
            f"are {SUPPORTED_FORMAT_TYPES}"
        )

    if format != "exllama":
        raise NotImplementedError(f"Exporting to format {format} is not supported yet.")

    # translate
    model, tokenizer = _create_model_and_tokenizer(model, tokenizer)
    _LOGGER.info(f"Translating state dict to {format} format.")
    translated_state_dict: Dict[str, Any] = translate_state_dict(
        state_dict=model.state_dict(), format=format
    )

    model.config.quantization_config = QuantizationConfig()
    _LOGGER.info(f"Added {format} quantization info to model.config")

    # save
    if save_dir is None:
        save_dir = Path.cwd() / f"{format}_model"

    save_dir: str = get_unique_dir_name(dir_name=save_dir)

    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        state_dict=translated_state_dict,
        save_dir=save_dir,
    )


def _create_model_and_tokenizer(
    model: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizerBase, str, None] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Create/infer model and tokenizer instances from the passed
    in model and tokenizer.

    :param model: The model to be exported, can also be
        path to a local model directory or a HuggingFace/SparseZoo stub
    :param tokenizer: The tokenizer associated with the model,
        can also be a HuggingFace/SparseZoo stub, if not passed in,
        it will be inferred from the model. An error will be raised if it
        cannot be inferred.
    :return A tuple of (model, tokenizer) instances. If both were
        passed into this function, they are returned as is.
        If tokenizer was not passed in, it is inferred from the
        model path/stub
    """
    if isinstance(tokenizer, str):
        # tokenizer from it's own path/stub
        tokenizer = SparseAutoTokenizer.from_pretrained(tokenizer)

    if tokenizer is None and isinstance(model, str):
        # tokenizer from model path/stub
        tokenizer = SparseAutoTokenizer.from_pretrained(model)

    if tokenizer is None:
        raise ValueError(
            "tokenizer not passed in and could not be inferred from model."
            "Please pass in a tokenizer."
        )

    if isinstance(model, str):
        model = SparseAutoModelForCausalLM.from_pretrained(model)

    return model, tokenizer


def save_checkpoint(
    model: PreTrainedModel,
    state_dict: Dict[Any, Any],
    save_dir: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    model.save_pretrained(
        save_directory=save_dir, state_dict=state_dict, safe_serialization=True
    )
    _LOGGER.info(f"Model and config saved to {save_dir}")

    if tokenizer:
        tokenizer.save_pretrained(save_directory=save_dir)
        _LOGGER.info(f"tokenizer saved to {save_dir}")


def translate_state_dict(
    state_dict: Dict[Any, Any], format: SUPPORTED_FORMAT_TYPES
) -> Dict[Any, Any]:
    """
    A utility function to translate the state dict to the specified format.

    :param state_dict: The state dict to be translated.
    :param format: The format to which the state dict should be translated.
    """
    if format == "exllama":
        return _translate_state_dict_exllama(state_dict=state_dict)


def _translate_state_dict_exllama(state_dict: Dict[str, Any]) -> Dict[Any, Any]:
    """
    Translate the state dict to the Exllama format.

    Changes made to quantized params in the passed state_dict:
    - weight tensor renamed to qweight, and the corresponding tensor
        value of shape [x, 8y] will be repacked to [x, y]
    - scale tensor renamed to scales, and the corresponding tensor
        value of shape [8x] will be reshaped to [1, 8x] and
        then repacked to [1, x]
    - zero_point tensor renamed to qzeros, and the corresponding tensor
        value of shape [x] will be reshaped to [1, x]
    - A g_idx tensor of shape [num_channels] will be added to the
        state_dict, this tensor will be filled with zeros
    - All fake quantization parameters will be removed from the state_dict




    :param state_dict: The model state dict to be translated.
    :return: The translated state dict compatible with Exllama.
    """

    transformations = (
        transform_names,
        add_tensors,
        transform_tensors,
        remove_unwanted_tensors,
        convert_fp32_to_fp16,
    )
    state_dict_copy = {}
    for transformation in transformations:
        state_dict_copy: Dict[str, Tensor] = transformation(
            state_dict=state_dict_copy or state_dict
        )

    return state_dict_copy


@dataclass
class QuantizationConfig:
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=-1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=False)
    sym: bool = field(default=True)
    is_marlin_format: bool = field(default=False)

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "is_marlin_format": self.is_marlin_format,
            "quant_method": "gptq",
        }
