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
import inspect
import logging
import os
from collections import OrderedDict
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple, Union

import requests
import torch
import transformers
from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy

from huggingface_hub import HUGGINGFACE_CO_URL_HOME, hf_hub_download
from sparseml.export.helpers import ONNX_MODEL_NAME
from sparsezoo import Model, setup_model


_LOGGER = logging.getLogger(__name__)


__all__ = [
    "RECIPE_NAME",
    "save_zoo_directory",
    "detect_last_checkpoint",
    "TaskNames",
    "is_transformer_model",
    "resolve_sequence_length",
    "ALL_TASK_NAMES",
    "create_fake_dataloader",
    "POSSIBLE_TOKENIZER_FILES",
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
POSSIBLE_TOKENIZER_FILES = {
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
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
            model = (
                model_args.model
                if hasattr(model_args, "model")
                else model_args.model_name_or_path
            )
            if os.path.isdir(model):
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


def resolve_recipe(
    recipe: Union[str, Path, None], model_path: Union[str, Path]
) -> Union[str, None]:
    """
    Resolve the recipe to apply to the model.
    :param recipe: the recipe to apply to the model.
        It can be one of the following:
        - None
            This means that we are not either not applying
            any recipe and allowing the model to potentially
            infer the appropriate pre-existing recipe
            from the model_path
        - a path to the recipe file
            This can be a string or Path object pointing
            to a recipe file. If the specified recipe file
            is different from the potential pre-existing
            recipe for that model (stored in the model_path),
            the function will raise an warning
        - name of the recipe file (e.g. "recipe.yaml")
            Recipe file name specific is assumed to be stored
            in the model_path
        - a string containing the recipe
            Needs to adhere to the SparseML recipe format

    :param model_path: the path to the model to load.
        It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model id
        - SparseZoo stub

    :return: the resolved recipe
    """

    if recipe is None:
        return infer_recipe_from_model_path(model_path)

    elif os.path.isfile(recipe):
        # recipe is a path to a recipe file
        return resolve_recipe_file(recipe, model_path)

    elif os.path.isfile(os.path.join(model_path, recipe)):
        # recipe is a name of a recipe file
        recipe = os.path.join(model_path, recipe)
        return resolve_recipe_file(recipe, model_path)

    elif isinstance(recipe, str):
        # recipe is a string containing the recipe
        _LOGGER.debug(
            "Applying the recipe string directly to the model, without "
            "checking for a potential existing recipe in the model_path."
        )
        return recipe

    _LOGGER.info(
        "No recipe requested and no default recipe "
        f"found in {model_path}. Skipping recipe resolution."
    )
    return None


def infer_recipe_from_model_path(model_path: Union[str, Path]) -> Optional[str]:
    """
    Infer the recipe from the model_path.
    :param model_path: the path to the model to load.
        It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model id
        - SparseZoo stub
    :return the path to the recipe file if found, None otherwise
    """
    model_path = model_path.as_posix() if isinstance(model_path, Path) else model_path

    if os.path.isdir(model_path) or os.path.isfile(model_path):
        # model_path is a local path to the model directory or model file
        # attempting to find the recipe in the model_directory
        model_path = (
            os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
        )
        recipe = os.path.join(model_path, RECIPE_NAME)
        if os.path.isfile(recipe):
            _LOGGER.info(f"Found recipe in the model_path: {recipe}")
            return recipe
        _LOGGER.debug(f"No recipe found in the model_path: {model_path}")
        return None

    elif model_path.startswith("zoo:"):
        # model_path is a sparsezoo stub
        return recipe_from_sparsezoo_stub(stub=model_path)

    recipe = recipe_from_huggingface_model_id(model_path)[0]

    if recipe is None:
        _LOGGER.info("Failed to infer the recipe from the model_path")
    return recipe


def recipe_from_huggingface_model_id(
    model_path: str, recipe_name: str = RECIPE_NAME
) -> Tuple[Optional[str], bool]:
    """
    Attempts to download the recipe from the huggingface model id.

    :param model_path: Assumed to be the huggingface model id.
        If it is not, this function will return None.
    :param recipe_name: The name of the recipe file to download.
        Defaults to RECIPE_NAME.
    :return: tuple:
        - the path to the recipe file if found, None otherwise
        - True if model_path is a valid huggingface model id, False otherwise
    """
    model_id = os.path.join(HUGGINGFACE_CO_URL_HOME, model_path)
    request = requests.get(model_id)
    if not request.status_code == 200:
        _LOGGER.debug(
            "model_path is not a valid huggingface model id. "
            "Skipping recipe resolution."
        )
        return None, False

    _LOGGER.info(
        "model_path is a huggingface model id. "
        "Attempting to download recipe from "
        f"{HUGGINGFACE_CO_URL_HOME}"
    )
    try:
        _LOGGER.info(
            f"Found recipe: {recipe_name} for model id: "
            f"{model_path}. Downloading..."
        )
        recipe = hf_hub_download(repo_id=model_path, filename=recipe_name)
    except Exception as e:
        _LOGGER.info(
            f"Unable to to find recipe {recipe_name} "
            f"for model id: {model_path}: {e}. "
            "Skipping recipe resolution."
        )
        recipe = None
    return recipe, True


def recipe_from_sparsezoo_stub(
    stub: str, recipe_file_name: Optional[str] = None
) -> Optional[str]:
    """
    Attempts to find the recipe for the sparsezoo stub.

    :param stub: The sparsezoo stub to find the recipe for
    :param recipe_file_name: The name of the recipe file to find.
        If None, the default recipe will be returned. Default: None
    :return: The path to the recipe file if found, None otherwise
    """
    if recipe_file_name is None:
        recipe = Model(stub).recipes.default.path
        _LOGGER.info(f"Found recipe: {recipe}")
        return recipe
    else:
        for recipe in Model(stub).recipes.recipes:
            if recipe.name == recipe_file_name:
                recipe = recipe.path
                _LOGGER.info(f"Found recipe: {recipe}")
                return recipe
        _LOGGER.warning(
            f"Unable to find recipe: {recipe_file_name} " f"for sparsezoo stub: {stub}."
        )
        return None


def resolve_recipe_file(
    requested_recipe: Union[str, Path], model_path: Union[str, Path]
) -> Union[str, Path, None]:
    """
    Given the requested recipe and the model_path, return the path to the recipe file.

    :param requested_recipe. Is a full path to the recipe file
    :param model_path: the path to the model to load.
        It can be one of the following:
        - a path to the model directory
        - a path to the model file
        - Hugging face model id
        - SparseZoo stub
    :return the path to the recipe file if found, None otherwise
    """
    # preprocess arguments so that they are all strings
    requested_recipe = (
        requested_recipe.as_posix()
        if isinstance(requested_recipe, Path)
        else requested_recipe
    )
    model_path = model_path.as_posix() if isinstance(model_path, Path) else model_path
    model_path = (
        os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    )

    if not os.path.isdir(model_path):
        # pathway for model_path that is not a directory
        if model_path.startswith("zoo:"):
            default_recipe = recipe_from_sparsezoo_stub(model_path)
        else:
            default_recipe, model_exists = recipe_from_huggingface_model_id(model_path)
            if not model_exists:
                raise ValueError(f"Unrecognized model_path: {model_path}")

        if not default_recipe == requested_recipe and default_recipe is not None:
            _LOGGER.warning(
                f"Attempting to apply recipe: {requested_recipe} "
                f"to the model at: {model_path}, "
                f"but the model already has a recipe: {default_recipe}. "
                f"Using {requested_recipe} instead."
            )
        return requested_recipe

    # pathway for model_path that is a directory
    default_recipe = os.path.join(model_path, RECIPE_NAME)
    default_recipe_exists = os.path.isfile(default_recipe)
    default_and_request_recipes_identical = os.path.samefile(
        default_recipe, requested_recipe
    )

    if (
        default_recipe_exists
        and requested_recipe
        and not default_and_request_recipes_identical
    ):
        _LOGGER.warning(
            f"Attempting to apply recipe: {requested_recipe} "
            f"to the model located in {model_path}, "
            f"but the model already has a recipe stored as {default_recipe}. "
            f"Using {requested_recipe} instead."
        )

    elif not default_recipe_exists and requested_recipe:
        _LOGGER.warning(
            f"Attempting to apply {requested_recipe} "
            f"to the model located in {model_path}."
            "However, it is expected that the model "
            f"has its target recipe stored as {default_recipe}."
            "Applying any recipe before the target recipe may "
            "result in unexpected behavior."
            f"Applying {requested_recipe} nevertheless."
        )

    elif default_recipe_exists:
        _LOGGER.info(f"Using the default recipe: {requested_recipe}")

    return requested_recipe


def create_fake_dataloader(
    model: torch.nn.Module,
    tokenizer: transformers.AutoTokenizer,
    num_samples: int,
) -> Tuple[Iterable[OrderedDictType[str, torch.Tensor]], List[str]]:
    """
    Creates fake transformers dataloader for the model, based on the model's
    forward signature.

    :param model: The model to create the dataloader for
    :param tokenizer: The tokenizer to use for the dataloader
    :param num_samples: The number of fake samples in the dataloader
    :return: The data loader (iterable) and the input names for the model
    """

    forward_args_spec = inspect.getfullargspec(model.__class__.forward)
    inputs = tokenizer(
        "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
    ).data
    fake_inputs = OrderedDict(
        [
            (input_key, inputs[input_key][0].reshape(1, -1))
            for input_key in forward_args_spec.args
            if input_key in inputs
        ]
    )
    data_loader = (fake_inputs for _ in range(num_samples))
    input_names = list(fake_inputs.keys())
    return data_loader, input_names


def fetch_recipe_path(target: str):
    """
    Fetches the recipe path for the given target.
    This method will also download the recipe if it is not
    already downloaded.

    Takes care of three scenarios:
    1. target is a local path to a model directory
        (looks for recipe.yaml in the directory)
    2. target is a SparseZoo stub (downloads and
        returns the path to the default recipe)
    3. target is a HuggingFace stub (downloads and
        returns the path to the default recipe)

    :param target: The target to fetch the recipe path for
        can be a local path, SparseZoo stub, or HuggingFace stub
    :return: The path to the recipe for the target
    """
    DEFAULT_RECIPE_NAME = "recipe.yaml"
    if Path(target).exists():
        # target is a local path
        potential_recipe_path = Path(target) / DEFAULT_RECIPE_NAME
        return str(potential_recipe_path) if potential_recipe_path.exists() else None

    # Recipe must be downloaded

    recipe_path = None
    if target.startswith("zoo:"):
        # target is a SparseZoo stub
        sparsezoo_model = Model(source=target)
        with suppress(Exception):
            # suppress any errors if the recipe is not found on SparseZoo
            recipe_path = sparsezoo_model.recipes.default().path
        return recipe_path

    # target is a HuggingFace stub
    with suppress(Exception):
        # suppress any errors if the recipe is not found on HuggingFace
        recipe_path = hf_hub_download(repo_id=target, filename=DEFAULT_RECIPE_NAME)

    return recipe_path
