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
import collections
import inspect
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint

from sparsezoo import setup_model


_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "RECIPE_NAME",
    "save_zoo_directory",
    "detect_last_checkpoint",
    "TaskNames",
    "create_dummy_inputs",
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


RECIPE_NAME = "recipe.yaml"
# TODO: To import MODEL_ONNX_NAME after the rebase
MODEL_ONNX_NAME = "model.onnx"
MANDATORY_DEPLOYMENT_FILES = {
    MODEL_ONNX_NAME,
    "tokenizer_config.json",
    "config.json",
}
NLG_TOKENIZER_FILES = {"special_tokens_map.json", "vocab.json", "merges.txt"}
OPTIONAL_DEPLOYMENT_FILES = {"tokenizer.json", "tokenizer.model"}


def is_transformer_model(source_path: Union[Path, str]) -> bool:
    """
    :param source_path: The path to the model
    :return: Whether the model is a transformers model or not
    """
    # make sure that the path is a directory and contains
    # the EXPECTED_TRANSFORMER_FILES
    if not os.path.isdir(source_path):
        raise ValueError(f"Path {source_path} is not a valid directory")
    expected_files = MANDATORY_DEPLOYMENT_FILES.difference({MODEL_ONNX_NAME})
    return expected_files.issubset(os.listdir(source_path))


def create_dummy_inputs(
    model: Any, tokenizer: AutoTokenizer, batch_size: int = 1, type: str = "pt"
) -> Dict[str, Union["torch.Tensor", numpy.ndarray]]:  # noqa 821
    """
    Create dummy inputs for the model to be exported to ONNX.
    The inputs are created by the tokenizer and then rearranged to match the order
    of inputs expected by the model's forward function.

    :param model: The pytorch model
    :param tokenizer: The tokenizer used to create the inputs
    :param batch_size: The batch size of the inputs
    :param type: The type of the inputs, either "pt" for pytorch tensors or "np" for
        numpy arrays
    :return: The dummy inputs as a dictionary of {input_name: input_value}
    """
    if type not in ["pt", "np"]:
        raise ValueError(f"Type of inputs must be one of ['pt', 'np'], got {type}")

    if not hasattr(model, "forward"):
        raise ValueError(
            f"Model: {model} is expected to have a forward function, but it does not"
        )

    inputs: Dict[str, Union["torch.Tensor", numpy.ndarray]] = tokenizer(  # noqa 821
        "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
    ).data

    # Rearrange inputs' keys to match those defined by model forward function, which
    # defines how the order of inputs is determined in the exported model
    forward_args_spec = inspect.getfullargspec(model.__class__.forward)

    # Drop inputs that were added by the tokenizer and are not expected by the model
    dropped = [
        input_key
        for input_key in inputs.keys()
        if input_key not in forward_args_spec.args
    ]
    if dropped:
        _LOGGER.warning(
            "The following inputs were not present in the model forward function "
            f"and therefore dropped from ONNX export: {dropped}"
        )

    # Rearrange inputs so that they all have shape (batch_size, tokenizer.max_length)
    inputs = collections.OrderedDict(
        [
            (
                func_input_arg_name,
                inputs[func_input_arg_name][0].reshape(
                    batch_size, tokenizer.model_max_length
                ),
            )
            for func_input_arg_name in forward_args_spec.args
            if func_input_arg_name in inputs
        ]
    )

    return inputs


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


def detect_last_checkpoint(training_args: "TrainingArguments"):  # noqa 821
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
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
