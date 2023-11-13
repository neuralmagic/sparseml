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
# TODO: Add docstrings

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, Optional

from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from sparseml.transformers.sparsification import Trainer
from src.sparseml.transformers.utils.model import TransformerModelsRegistry


__all__ = ["initialize_transformer_model"]

_LOGGER = logging.getLogger(__name__)


@dataclass
class ForceCPUTrainingArguments(TrainingArguments):
    @property
    def place_model_on_device(self):
        # TODO: Observe how this setting influences memory consumption
        # The property governs whether or not to automatically place
        # the model on the device. Setting to False ensures that the
        # model remains in CPU during ONNX export
        return False


def initialize_transformer_model(
    model_path: Union[str, Path],
    sequence_length: int,
    task: str,
    trust_remote_code: bool = False,
    **config_args,
):

    config = initialize_config(model_path, trust_remote_code, **config_args)
    tokenizer = initialize_tokenizer(model_path, task, sequence_length)
    model = TransformerModelsRegistry.load_from_registry(task)(
        **dict(
            model_name_or_path=model_path,
            model_type="model",
            config=config,
            trust_remote_code=trust_remote_code,
        )
    )
    model.train()
    trainer = initialize_trainer(model, model_path)
    model.eval()

    _LOGGER.info(f"Loaded model, trainer config, and tokenizer from {model_path}")
    return model, trainer, config, tokenizer


def initialize_trainer(model: Any, model_path: Union[str, Path]) -> Trainer:
    training_args = TrainingArguments(output_dir=os.path.dirname(model_path))
    trainer = Trainer(
        model=model,
        args=training_args,
        model_state_path=model_path,
        # TODO: Do we need eval_dataset?
        # eval_dataset=eval_dataset,
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
    model_path: Union[str, Path], task: str, sequence_length: Optional[int] = None,
) -> AutoTokenizer:

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length
    )
    if task == "text-generation":
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
