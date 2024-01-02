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
import re
from pathlib import Path
from typing import Any, Optional, Union

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments

from sparseml.optim import parse_recipe_variables
from sparseml.transformers.sparsification import Trainer
from sparseml.transformers.utils.helpers import RECIPE_NAME, TaskNames
from sparseml.transformers.utils.load_task_model import load_task_model


__all__ = [
    "initialize_sparse_model",
    "initialize_tokenizer",
    "initialize_trainer",
    "initialize_config",
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
    model_path: Union[str, Path], sequence_length: int, task: str, **tokenizer_args
) -> AutoTokenizer:
    """
    Initialize a tokenizer from a given path

    :param model_path: the path to the model to load
    :param sequence_length: the sequence length to use for the tokenizer
    :param task: the task to load the tokenizer for
    :return: the loaded tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length, **tokenizer_args
    )
    if task in TaskNames.text_generation.value:
        # for generative transformers, we might
        # need to set the pad token to the eos token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def initialize_sparse_model(
    model_path: Union[str, Path],
    task: str,
    config: AutoConfig,
    trust_remote_code: bool = False,
    recipe: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    **model_kwargs,
) -> AutoModel:
    """
    Initialize a sparse model from a given path. This will:
    1. Load the dense model from the given path, given
       the task name, config and model_kwargs
    2.(Optional) Apply the recipe to the model.
      If no recipe is given, will look for a recipe
      in the model_path
    3 (Optional) Move the model to the specified device

    Example usage:
    ```python
     model_path = ... # path to the model
     task = ... # the task to load the model for
        e.g "text-generation" or "question-answering"

     config = initialize_config(model_path=model_path,
                                trust_remote_code=True)

     model = initialize_sparse_model(
        model_path=model_path,
        task=self.task,
        config=config,
        )
    ```

    :param model_path: the path to the model to load
    :param task: the task to load the model for
    :param config: the config to use for the model
    :param trust_remote_code: True to trust remote code when loading the model,
        False otherwise
    :param recipe: the recipe to apply to the model. If None, will look for a recipe
        in the model_path
    :param device: the device to load the model on. If None, will load on CPU
    :return: the loaded model
    """
    recipe = resolve_recipe_application(recipe, model_path)

    model = load_task_model(
        task=task,
        model_path=model_path,
        config=config,
        trust_remote_code=trust_remote_code,
        recipe=recipe,
        **model_kwargs,
    )
    if device:
        # if device is a list of devices, then we assume we want to use multiple gpus
        # (wrap the model in a DataParallel) e.g. device = 'cuda:0,1,...'
        use_multiple_gpus = re.match(r"cuda:\d+,(\d+)*", device)
        model = torch.nn.DataParallel(model) if use_multiple_gpus else model.to(device)
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
    # TODO: add here support for v2 trainer
    # also add initialize_dataset function that will merge v1 and v2 functions

    training_args = TrainingArguments(
        output_dir=os.path.dirname(model_path), use_cpu=(model.device.type == "cpu")
    )

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


def resolve_recipe_application(
    recipe: Union[str, Path, None], model_path: Union[str, Path]
) -> Union[str, Path, None]:
    """
    Resolve the recipe to apply to the model.
    :param recipe: the recipe to apply to the model.
        It can be one of the following:
        - None (no recipe will be applied or the
            default recipe will be applied if exists. Default recipe
            is assumed to be stored in the model_path and named RECIPE_NAME)
        - a path to the recipe file
        - name of the recipe file (e.g. "recipe.yaml")
            (assumed to be stored in the model_path instead
            of RECIPE_NAME)
        - a string containing the recipe
    :param model_path: the path to the model to load
    :return: the resolved recipe
    """

    if recipe is None:
        # if recipe is None -> still look for recipe.yaml in the model_path
        recipe = os.path.join(model_path, RECIPE_NAME)
        if os.path.isfile(recipe):
            return recipe

    elif os.path.isfile(recipe):
        # recipe is a path to a recipe file
        return _resolve_recipe_file(recipe, model_path)

    elif os.path.isfile(os.path.join(model_path, recipe)):
        # recipe is a name of a recipe file
        recipe = os.path.join(model_path, recipe)
        return _resolve_recipe_file(recipe, model_path)
    elif isinstance(recipe, str):
        # recipe is a string containing the recipe
        _LOGGER.debug(
            "Applying the recipe string directly to the model, without "
            "checking for a potential existing recipe in the model_path."
        )
        return recipe

    _LOGGER.info(
        "No recipe requested and no default recipe "
        f"found in {model_path}. Skipping recipe application."
    )
    return None


def _resolve_recipe_file(
    requested_recipe: Union[str, Path], model_path: Union[str, Path]
) -> Union[str, Path, None]:
    default_recipe = os.path.join(model_path, RECIPE_NAME)
    default_recipe_exists = os.path.isfile(default_recipe)
    default_and_request_recipes_identical = default_recipe == requested_recipe

    if (
        default_recipe_exists
        and requested_recipe
        and not default_and_request_recipes_identical
    ):
        _LOGGER.warning(
            f"Attempting to apply {requested_recipe} "
            f"to the model located in {model_path}, "
            f"but the model already has a recipe stored in {default_recipe}. "
            f"Using {requested_recipe} instead."
        )
        return requested_recipe

    elif (
        not default_recipe_exists
        and requested_recipe
        and not default_and_request_recipes_identical
    ):
        _LOGGER.warning(
            f"Attempting to apply {requested_recipe} "
            f"to the model located in {model_path}."
            "However, it is expected that the model "
            f"has it's target recipe stored as {default_recipe}."
            "Applying any recipe before the target recipe may "
            "result in unexpected behavior."
            f"Applying {requested_recipe} nevertheless."
        )
        return requested_recipe

    elif default_recipe_exists:
        _LOGGER.info(f"Applying the default recipe: {default_recipe}")
        return default_recipe


def _parse_data_args(data_args):
    try:
        return parse_recipe_variables(data_args)
    except ValueError as parse_error:
        message = str(parse_error).replace("recipe_args", "data_args")
        if "recipe variables" in message:
            message = message.replace("recipe variables", "data_args")
        raise ValueError(message)
