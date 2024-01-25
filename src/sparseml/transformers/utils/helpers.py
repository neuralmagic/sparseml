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
Helper variables and functions for integrating SparseML with huggingface/transformers
flows
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint

from sparseml.export.helpers import ONNX_MODEL_NAME
from sparsezoo import setup_model


_LOGGER = logging.getLogger(__name__)


__all__ = [
    "RECIPE_NAME",
    "save_zoo_directory",
    "detect_last_checkpoint",
    "TaskNames",
    "is_transformer_model",
    "resolve_sequence_length",
    "ALL_TASK_NAMES",
]


class TaskNames(Enum):
    mlm = {"masked-language-modeling", "mlm"}
    qa = {"question-answering", "qa"}
    token_classification = {"token-classification", "ner"}
    text_classification = {
        "text-classification",
        "sentiment-analysis",
        "sequence-classification",
        "glue",
    }
    text_generation = {"text-generation"}


ALL_TASK_NAMES = list(set.union(*[task_names.value for task_names in TaskNames]))
ONNX_MODEL_NAME_INTERMEDIATE = "model-orig.onnx"
RECIPE_NAME = "recipe.yaml"
MANDATORY_DEPLOYMENT_FILES = {
    ONNX_MODEL_NAME,
    "tokenizer_config.json",
    "config.json",
}
OPTIONAL_DEPLOYMENT_FILES = {"tokenizer.json", "tokenizer.model"}
NLG_MANDATORY_DEPLOYMENT_FILES = {"special_tokens_map.json"}
NLG_OPTIONAL_DEPLOYMENT_FILES = {
    ONNX_MODEL_NAME_INTERMEDIATE,
    "vocab.json",
    "merges.txt",
}


def remove_past_key_value_support_from_config(config: AutoConfig) -> AutoConfig:
    """
    Modify config of the causal language model so that it turns off the
    past key value support. This means that the model initialized from
    this config will not take past key values as input and will not output
    past key values.
    """
    # not take past_key_values as input
    config.is_decoder = True
    # whether to use past key values an input
    config.use_past = False
    # whether to output past key values
    config.use_cache = False
    return config


def is_transformer_model(source_path: Union[Path, str]) -> bool:
    """
    :param source_path: The path to the model
    :return: Whether the model is a transformers model or not
    """
    if not os.path.isdir(source_path):
        raise ValueError(f"Path {source_path} is not a valid directory")
    expected_files = MANDATORY_DEPLOYMENT_FILES.difference({ONNX_MODEL_NAME})
    return expected_files.issubset(os.listdir(source_path))


def save_zoo_directory(
    output_dir: str,
    training_outputs_dir: str,
    logs_path: Optional[str] = None,
):
    """
    Takes the `training_outputs_dir`
    (the directory where the pipeline saves its training artifacts),
    and saves the training artifacts to `output_dir` as a sparsezoo Model class object.

    :param output_dir: The output path where the artifacts are saved
        (adhering to the structure of sparsezoo Model class object)
    :param training_outputs_dir: The path to the existing directory
        with the saved training artifacts
    :param logs_path: Optional directory where the training logs reside
    """
    for root_file in ["sample-inputs", "sample-outputs"]:
        root_file_path = os.path.join(training_outputs_dir, root_file)
        if not os.path.exists(root_file_path):
            _LOGGER.warning(
                f"File {root_file_path} missing. To create this file, "
                "make sure that the export script is being ran with"
                "`--num_export_samples` argument."
            )
    for root_file in ["model.onnx", "deployment"]:
        root_file_path = os.path.join(training_outputs_dir, root_file)
        if not os.path.exists(root_file_path):
            raise ValueError(
                f"File {root_file_path} missing. To create this file, "
                "make sure that the `export` script (for exporting "
                "transformer models) has been evoked."
            )

    setup_model(
        output_dir=output_dir,
        training=os.path.join(training_outputs_dir, "training"),
        deployment=os.path.join(training_outputs_dir, "deployment"),
        onnx_model=os.path.join(training_outputs_dir, "model.onnx"),
        sample_inputs=os.path.join(training_outputs_dir, "sample-inputs"),
        sample_outputs=os.path.join(training_outputs_dir, "sample-outputs"),
        model_card=os.path.join(training_outputs_dir, "model.md"),
        logs=logs_path,
        sample_labels=None,
        sample_originals=None,
        analysis=None,
        benchmarks=None,
        eval_results=None,
        recipes=None,
    )
    _LOGGER.info(f"Created sparsezoo Model directory locally in {output_dir}")


def detect_last_checkpoint(
    training_args: "TrainingArguments",  # noqa 821
    model_args: Optional["ModelArguments"] = None,  # noqa 821
):
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if training_args.run_stages and model_args is not None:
            if os.path.isdir(model_args.model_name_or_path):
                last_checkpoint = get_last_checkpoint(model_args.model_name_or_path)
        if last_checkpoint is None and (len(os.listdir(training_args.output_dir)) > 0):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already "
                "exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            _LOGGER.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change  the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


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
    _LOGGER.debug(
        f"Using default sequence length of {sequence_length} "
        "(inferred from HF transformers config) "
    )
    return sequence_length


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
