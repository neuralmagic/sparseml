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
Helper functions and script for exporting a trained transformers model to an ONNX file
for use with engines such as DeepSparse

script accessible from sparseml.transformers.export_onnx

command help:
usage: export.py [-h] --task TASK --model_path MODEL_PATH
                 [--sequence_length SEQUENCE_LENGTH]
                 [--convert_qat CONVERT_QAT]
                 [--finetuning_task FINETUNING_TASK]
                 [--onnx_file_name ONNX_FILE_NAME]

Export a trained transformers model to an ONNX file

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           task to create the model for. i.e. mlm, qa, glue, ner
  --model_path MODEL_PATH
                        Path to directory where model files for weights, config,
                        and tokenizer are stored
  --sequence_length SEQUENCE_LENGTH
                        Sequence length to use. Default is 384. Can be overwritten
                        later
  --convert_qat CONVERT_QAT
                        Set True to convert QAT graph exports to fully quantized.
                        Default is True
  --finetuning_task FINETUNING_TASK
                        optional finetuning task for text classification and token
                        classification exports
  --onnx_file_name ONNX_FILE_NAME
                        name for exported ONNX file in the model directory. Default
                        and reccomended value for pipeline compatibility is
                        'model.onnx'

example usage:
sparseml.transformers.export_onnx \
  --task question-answering \
  --model_path /PATH/TO/SPARSIFIED/MODEL/DIRECTORY \
  --sequence_length 128
"""

import argparse
import logging
import os
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.tokenization_utils_base import PaddingStrategy

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import export_onnx
from sparseml.transformers.utils.helpers import RECIPE_NAME


__all__ = ["export_transformer_to_onnx"]


_LOGGER = logging.getLogger(__name__)
_TASK_TO_CONSTRUCTOR = {
    # language modeling
    "mlm": AutoModelForMaskedLM,
    "masked-language-modeling": AutoModelForMaskedLM,
    # question answering
    "qa": AutoModelForQuestionAnswering,
    "question-answering": AutoModelForQuestionAnswering,
    # GLUE
    "glue": AutoModelForSequenceClassification,
    "sequence-classification": AutoModelForSequenceClassification,
    "sentiment-analysis": AutoModelForSequenceClassification,
    "text-classification": AutoModelForSequenceClassification,
    # token classification
    "ner": AutoModelForTokenClassification,
    "token-classification": AutoModelForTokenClassification,
}


def export_transformer_to_onnx(
    task: str,
    model_path: str,
    sequence_length: int = 384,
    convert_qat: bool = True,
    finetuning_task: Optional[str] = None,
    onnx_file_name: str = "model.onnx",
) -> str:
    """
    Exports the saved transformers file to ONNX at batch size 1 using
    the given model path weights, config, and tokenizer

    :param task: task to create the model for. i.e. mlm, qa, glue, ner
    :param model_path: path to directory where model files, tokenizers,
        and configs are saved. ONNX export will also be written here
    :param sequence_length: model sequence length to use for export
    :param convert_qat: set True to convert a QAT model to fully quantized
        ONNX model. Default is True
    :param finetuning_task: optional string finetuning task for text classification
        and token classification exports
    :param onnx_file_name: name to save the exported ONNX file as. Default
        is model.onnx. Note that when loading a model directory to a deepsparse
        pipeline, it will look only for 'model.onnx'
    :return: path to the exported ONNX file
    """
    if task.lower() not in _TASK_TO_CONSTRUCTOR:
        raise ValueError(
            f"task {task} unsupported for export_transformer_to_onnx. Supported "
            f"tasks include {list(_TASK_TO_CONSTRUCTOR.keys())}"
        )
    auto_model_constructor = _TASK_TO_CONSTRUCTOR[task.lower()]

    if not os.path.isdir(model_path):
        raise ValueError(
            "model_path must be a directory that contains the trained transformer "
            f"files. {model_path} is not a directory"
        )

    # load config and tokenizer
    config_args = {"finetuning_task": finetuning_task} if finetuning_task else {}
    config = AutoConfig.from_pretrained(model_path, **config_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length
    )

    # load model
    model = auto_model_constructor.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
    )

    # apply recipe if exists before loading model weights
    recipe_path = os.path.join(model_path, RECIPE_NAME)
    if os.path.isfile(recipe_path):
        ScheduledModifierManager.from_yaml(recipe_path).apply(model)
    else:
        _LOGGER.warning(f"recipe not found under {recipe_path}")

    # load weights
    state_dict = torch.load(os.path.join(model_path, WEIGHTS_NAME))
    model.load_state_dict(state_dict)

    # create fake model input
    inputs = tokenizer(
        "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
    ).data  # Dict[Tensor]

    # run export
    onnx_file_path = os.path.join(model_path, onnx_file_name)
    export_onnx(
        model,
        inputs,
        onnx_file_path,
        convert_qat=convert_qat,
    )

    return onnx_file_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained transformers model to an ONNX file"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to create the model for. i.e. mlm, qa, glue, ner",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help=(
            "Path to directory where model files for weights, config, and "
            "tokenizer are stored"
        ),
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=384,
        help="Sequence length to use. Default is 384. Can be overwritten later",
    )
    parser.add_argument(
        "--convert_qat",
        type=bool,
        default=True,
        help=(
            "Set True to convert QAT graph exports to fully quantized. Default is True"
        ),
    )
    parser.add_argument(
        "--finetuning_task",
        type=str,
        default=None,
        help=(
            "Optional finetuning task for text classification and token "
            "classification exports"
        ),
    )
    parser.add_argument(
        "--onnx_file_name",
        type=str,
        default="model.onnx",
        help=(
            "Name for exported ONNX file in the model directory. "
            "Default and reccomended value for pipeline compatibility is 'model.onnx'"
        ),
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    _LOGGER.info(f"Exporting {args.model_path} to ONNX")
    onnx_path = export_transformer_to_onnx(
        task=args.task,
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        convert_qat=args.convert_qat,
        finetuning_task=args.finetuning_task,
        onnx_file_name=args.onnx_file_name,
    )
    _LOGGER.info(f"Model exported to: {onnx_path}")


if __name__ == "__main__":
    main()
