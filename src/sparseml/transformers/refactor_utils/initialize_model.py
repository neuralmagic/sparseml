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
from pathlib import Path
from typing import Union

from transformers import AutoConfig, AutoTokenizer

from src.sparseml.transformers.utils.model import TransformerModelsRegistry


__all__ = ["initialize_transformer_model"]

_LOGGER = logging.getLogger(__name__)


def initialize_transformer_model(
    model_path: Union[str, Path],
    sequence_length: int,
    task: str,
    trust_remote_code: bool = False,
    **config_args,
):

    config = initialize_config(model_path, trust_remote_code, **config_args)
    tokenizer = initialize_tokenizer(model_path, sequence_length, task)
    model = TransformerModelsRegistry.load_from_registry(task)(
        **dict(
            model_name_or_path=model_path,
            model_type="model",
            config=config,
            trust_remote_code=trust_remote_code,
        )
    )
    model.train()
    trainer = None
    _LOGGER.info(f"loaded model, config, and tokenizer from {model_path}")
    return model, config, tokenizer


def initialize_config(
    model_path: Union[str, Path], trust_remote_code: bool = False, **config_args
):
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
    if task == "text-generation":
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
