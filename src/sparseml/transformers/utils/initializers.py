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
Functionality for initializing a transformer model from a given path
"""

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Union

from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments

from sparseml.optim import parse_recipe_variables
from sparseml.transformers.sparsification import Trainer
from sparseml.transformers.utils.helpers import TaskNames
from sparseml.transformers.utils.load_task_model import load_task_model


__all__ = [
    "initialize_model",
    "initialize_tokenizer",
    "initialize_trainer",
    "initialize_config",
    "resolve_sequence_length",
]

_LOGGER = logging.getLogger(__name__)


def initialize_config(
    model_path: Union[str, Path], trust_remote_code: bool = False, **config_args
) -> AutoConfig:
    """
    Initialize a config from a given path

    :param model_path: the path to the model to load
    :param trust_remote_code: True to trust remote code when loading the model,
        False otherwise
    :param config_args: additional arguments to pass to the config
    :return: the loaded config
    """
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        **config_args,
    )
    return config


def initialize_tokenizer(
    model_path: Union[str, Path], sequence_length: int, task: str
) -> AutoTokenizer:
    """
    Initialize a tokenizer from a given path

    :param model_path: the path to the model to load
    :param sequence_length: the sequence length to use for the tokenizer
    :param task: the task to load the tokenizer for
    :return: the loaded tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length
    )
    if task in TaskNames.text_generation.value:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def initialize_model(
    model_path: Union[str, Path],
    task: str,
    config: AutoConfig,
    trust_remote_code: bool = False,
    device: Optional[str] = None,
) -> AutoModel:
    """
    Initialize a model from a given path

    :param model_path: the path to the model to load
    :param task: the task to load the model for
    :param config: the config to use for the model
    :param trust_remote_code: True to trust remote code when loading the model,
        False otherwise
    :param device: the device to load the model on. If None, will load on CPU
    :return: the loaded model
    """
    model = load_task_model(
        task=task,
        model_path=model_path,
        config=config,
        trust_remote_code=trust_remote_code,
    )
    if device:
        model.to(device)
    return model


def initialize_trainer(
    model: AutoModel,
    model_path: Union[str, Path],
    validation_dataset: Optional[Any] = None,
) -> Trainer:
    """
    Initialize a trainer. This will apply the structure dictated by
    any of the recipes stored in the model_path

    :param model: the model to initialize the trainer with
    :param model_path: the path to the model to load
    :param validation_dataset: the validation dataset to use for the trainer
    :return: the initialized trainer
    """

    training_args = TrainingArguments(output_dir=os.path.dirname(model_path))

    trainer = Trainer(
        model=model,
        args=training_args,
        model_state_path=model_path,
        eval_dataset=validation_dataset,
        recipe=None,
        recipe_args=None,
        teacher=None,
    )
    applied = trainer.apply_manager(epoch=math.inf, checkpoint=None)

    if not applied:
        _LOGGER.warning(
            f"No recipes were applied for {model_path}, "
            "check to make sure recipe(s) are stored in the model_path"
        )
    else:
        trainer.finalize_manager()
        num_stages = 0
        if trainer.manager:
            num_stages += trainer.manager.num_stages()
        if trainer.arch_manager:
            num_stages += trainer.arch_manager.num_stages()

        msg = (
            "an unstaged recipe"
            if num_stages == 1
            else f"a staged recipe with {num_stages} stages"
        )
        _LOGGER.info(f"Applied {msg} to the model at {model_path}")

    return trainer


def resolve_sequence_length(config: AutoConfig) -> int:
    """
    Resolve the sequence length from the config

    :param config: the config to resolve the sequence length from
    :return: the sequence length
    """
    if hasattr(config, "max_position_embeddings"):
        sequence_length = config.max_position_embeddings

    elif hasattr(config, "max_seq_len"):
        sequence_length = config.max_seq_len
    else:
        raise ValueError(
            "Could not infer a default sequence length "
            "from the HF transformers config. Please specify "
            "the sequence length with --sequence_length"
        )
    _LOGGER.info(
        f"Using default sequence length of {sequence_length} "
        "(inferred from HF transformers config) "
    )
    return sequence_length


def _parse_data_args(data_args):
    try:
        return parse_recipe_variables(data_args)
    except ValueError as parse_error:
        message = str(parse_error).replace("recipe_args", "data_args")
        if "recipe variables" in message:
            message = message.replace("recipe variables", "data_args")
        raise ValueError(message)
