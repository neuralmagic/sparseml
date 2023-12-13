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

import inspect
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from torch.nn import Module
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from sparseml.optim import parse_recipe_variables
from sparseml.transformers.sparsification import Trainer
from sparseml.transformers.utils.helpers import TaskNames
from sparseml.transformers.utils.load_task_dataset import load_task_dataset
from sparseml.transformers.utils.load_task_model import load_task_model


__all__ = ["create_model"]

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForceCPUTrainingArguments(TrainingArguments):
    @property
    def place_model_on_device(self):
        # The property governs whether to automatically place
        # the model on the device. Setting to False ensures that the
        # model remains in CPU during ONNX export
        return False


def create_model(
    model_path: Union[str, Path],
    task: str,
    sequence_length: Optional[int] = None,
    trust_remote_code: bool = False,
    data_args: Optional[Dict] = None,
    **config_args,
):

    config = initialize_config(model_path, trust_remote_code, **config_args)
    sequence_length = sequence_length or resolve_sequence_length(config)
    tokenizer = initialize_tokenizer(model_path, sequence_length, task)

    model = load_task_model(
        task=task,
        model_path=model_path,
        config=config,
        trust_remote_code=trust_remote_code,
    )
    data_args = _parse_data_args(data_args)

    validation_dataset = None
    if data_args:
        dataset = load_task_dataset(
            task=task,
            tokenizer=tokenizer,
            data_args=data_args,
            model=model,
            config=config,
        )
        validation_dataset = dataset.get("validation")

    model.train()
    trainer = initialize_trainer(model, model_path, validation_dataset)

    _LOGGER.info(f"Loaded model, trainer config, and tokenizer from {model_path}")
    return model, trainer, config, tokenizer, validation_dataset


def initialize_trainer(
    model: Any, model_path: Union[str, Path], validation_dataset
) -> Trainer:
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


def initialize_config(
    model_path: Union[str, Path], trust_remote_code: bool = False, **config_args
) -> AutoConfig:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        **config_args,
    )
    return config


def initialize_tokenizer(
    model_path: Union[str, Path], sequence_length: int, task: str
) -> AutoTokenizer:

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length
    )
    if task in TaskNames.text_generation.value:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def get_shared_tokenizer_src(student: Module, teacher: Optional[Module]) -> str:
    """
    Get a tokenizer source used for both student and teacher, assuming
    that they could be shared

    :param student: the student model
    :param teacher: the teacher model
    :return: the source for the tokenizer shared between teacher and model
    """

    if teacher is not None and teacher not in ("disable", "self"):
        student_forward_params = list(
            inspect.signature(student.forward).parameters.keys()
        )
        teacher_forward_params = list(
            inspect.signature(teacher.forward).parameters.keys()
        )
        diff = [p for p in student_forward_params if p not in teacher_forward_params]
        if diff:
            raise RuntimeError(
                "Teacher tokenizer cannot be used for student "
                f"due to missing args: {diff}"
            )
        src_model = teacher
    else:
        src_model = student
    return src_model.config._name_or_path


def _parse_data_args(data_args):
    try:
        return parse_recipe_variables(data_args)
    except ValueError as parse_error:
        message = str(parse_error).replace("recipe_args", "data_args")
        if "recipe variables" in message:
            message = message.replace("recipe variables", "data_args")
        raise ValueError(message)
