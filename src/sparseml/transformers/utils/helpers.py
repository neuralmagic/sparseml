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
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers.trainer_utils import get_last_checkpoint

from sparseml.export.helpers import ONNX_MODEL_NAME
from sparsezoo import setup_model


_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "RECIPE_NAME",
    "save_zoo_directory",
    "detect_last_checkpoint",
    "TaskNames",
    "is_transformer_model",
    "is_transformer_generative_model",
    "run_transformers_inference",
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
MANDATORY_DEPLOYMENT_FILES = {
    ONNX_MODEL_NAME,
    "tokenizer_config.json",
    "config.json",
}
NLG_TOKENIZER_FILES = {"special_tokens_map.json", "vocab.json", "merges.txt"}
OPTIONAL_DEPLOYMENT_FILES = {"tokenizer.json", "tokenizer.model"}


# TODO: Move this this functionality to export module once merged
def run_transformers_inference(
    inputs: Dict[str, Any], model: Optional[torch.nn.Module] = None
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """
    Run inference on a transformers model and return the inputs, labels and outputs

    :param inputs: The inputs to run inference on
    :param model: The model to run inference on (optional)

    :return: The inputs, labels and outputs
    """
    label = None  # transformers in general have no labels
    if model is None:
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        return inputs, label, None

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_vals = model(**inputs)
    inputs = {key: value.to("cpu") for key, value in inputs.items()}
    output = {
        name: torch.squeeze(val).detach().to("cpu") for name, val in output_vals.items()
    }
    return inputs, label, output


def is_transformer_model(source_path: Union[Path, str]) -> bool:
    """
    :param source_path: The path to the model
    :return: Whether the model is a transformers model or not
    """
    # make sure that the path is a directory and contains
    # the EXPECTED_TRANSFORMER_FILES
    if not os.path.isdir(source_path):
        raise ValueError(f"Path {source_path} is not a valid directory")
    expected_files = MANDATORY_DEPLOYMENT_FILES.difference({ONNX_MODEL_NAME})
    return expected_files.issubset(os.listdir(source_path))


def is_transformer_generative_model(source_path: Union[Path, str]) -> bool:
    """
    :param source_path: The path to the model
    :return: Whether the model is a transformers model or not
    """
    # make sure that the path is a directory and contains
    # the EXPECTED_TRANSFORMER_FILES
    if not os.path.isdir(source_path):
        raise ValueError(f"Path {source_path} is not a valid directory")
    expected_files = MANDATORY_DEPLOYMENT_FILES.union(NLG_TOKENIZER_FILES).difference(
        {ONNX_MODEL_NAME}
    )
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
