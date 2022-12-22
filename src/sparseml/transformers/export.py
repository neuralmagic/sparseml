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
Helper functions and script for exporting inference artifacts
to be used by inference engines such as DeepSparse.
The export incorporates:
- creating the deployment directory (the direct input to the DeepSparse
    inference pipeline)
- creating an ONNX file representing a trained transformers model

script accessible from sparseml.transformers.export_onnx

command help:
usage: sparseml.transformers.export_onnx [-h] --task TASK --model_path
                                         MODEL_PATH
                                         [--sequence_length SEQUENCE_LENGTH]
                                         [--no_convert_qat]
                                         [--finetuning_task FINETUNING_TASK]
                                         [--onnx_file_name ONNX_FILE_NAME]
                                         [--one_shot ONE_SHOT]

Export a trained transformers model to an ONNX file

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           Task to create the model for. i.e. mlm, qa, glue, ner
  --model_path MODEL_PATH
                        Path to directory where model files for weights,
                        config, and tokenizer are stored
  --sequence_length SEQUENCE_LENGTH
                        Sequence length to use. Default is 384. Can be
                        overwritten later
  --no_convert_qat      Set flag to not perform QAT to fully quantized
                        conversion after export
  --finetuning_task FINETUNING_TASK
                        Optional finetuning task for text classification and
                        token classification exports
  --onnx_file_name ONNX_FILE_NAME
                        Name for exported ONNX file in the model directory.
                        Default and recommended value for pipeline
                        compatibility is model.onnx
  --one_shot ONE_SHOT   local path or SparseZoo stub to a recipe that should
                        be applied in a one-shot manner before exporting
  --num_export_samples NUM_EXPORT_SAMPLES
                        Number of samples (inputs/outputs) to export
  --data_args DATA_ARGS
                        Valid json loadable args used to instantiate a
                        `DataTrainingArguments` instance while exporting
                        samples

example usage:
sparseml.transformers.export_onnx \
  --task question-answering \
  --model_path /PATH/TO/SPARSIFIED/MODEL/DIRECTORY \
  --sequence_length 128
"""

import argparse
import collections
import copy
import inspect
import logging
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Union

from torch.nn import Module
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

from sparseml.optim import parse_recipe_variables
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import export_onnx
from sparseml.transformers.sparsification import Trainer
from sparseml.transformers.utils import SparseAutoModel


__all__ = ["export_transformer_to_onnx", "load_task_model"]

MODEL_ONNX_NAME = "model.onnx"
DEPLOYMENT_FILES: List[str] = [
    MODEL_ONNX_NAME,
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
]

_LOGGER = logging.getLogger(__name__)


def load_task_model(task: str, model_path: str, config: Any) -> Module:
    if task == "masked-language-modeling" or task == "mlm":
        return SparseAutoModel.masked_language_modeling_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
        )

    if task == "question-answering" or task == "qa":
        return SparseAutoModel.question_answering_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
        )

    if (
        task == "sequence-classification"
        or task == "glue"
        or task == "sentiment-analysis"
        or task == "text-classification"
    ):
        return SparseAutoModel.text_classification_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
        )

    if task == "token-classification" or task == "ner":
        return SparseAutoModel.token_classification_from_pretrained(
            model_name_or_path=model_path,
            config=config,
            model_type="model",
        )

    raise ValueError(f"unrecognized task given of {task}")


def load_task_dataset(
    task: str, tokenizer, data_args: Dict[str, Any], model: Module, config=None
):
    """
    :param task: the task a dataset being loaded for
    :param tokenizer: the tokenizer to use for the dataset
    :param data_args: additional data args used to create a `DataTrainingArguments`
        instance for fetching the dataset
    """

    if task == "masked-language-modeling" or task == "mlm":
        from sparseml.transformers.masked_language_modeling import (
            DataTrainingArguments,
            get_tokenized_mlm_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        return get_tokenized_mlm_dataset(
            data_args=data_training_args, tokenizer=tokenizer
        )

    if task == "question-answering" or task == "qa":
        from sparseml.transformers.question_answering import (
            DataTrainingArguments,
            get_tokenized_qa_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        return get_tokenized_qa_dataset(
            data_args=data_training_args, tokenizer=tokenizer
        )

    if task == "token-classification" or task == "ner":
        from sparseml.transformers.token_classification import (
            DataTrainingArguments,
            get_tokenized_token_classification_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)
        return get_tokenized_token_classification_dataset(
            data_args=data_training_args, tokenizer=tokenizer, model=model
        )

    if (
        task == "sequence-classification"
        or task == "glue"
        or task == "sentiment-analysis"
        or task == "text-classification"
    ):

        from sparseml.transformers.text_classification import (
            DataTrainingArguments,
            get_tokenized_text_classification_dataset,
        )

        data_training_args = DataTrainingArguments(**data_args)

        return get_tokenized_text_classification_dataset(
            data_args=data_training_args,
            tokenizer=tokenizer,
            model=model,
            config=config,
        )

    raise ValueError(f"unrecognized task given of {task}")


def export_transformer_to_onnx(
    task: str,
    model_path: str,
    sequence_length: int = 384,
    convert_qat: bool = True,
    finetuning_task: Optional[str] = None,
    onnx_file_name: str = MODEL_ONNX_NAME,
    num_export_samples: int = 0,
    data_args: Optional[Union[Dict[str, Any], str]] = None,
    one_shot: Optional[str] = None,
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
    :param num_export_samples: number of samples (inputs/outputs) to export
    :param data_args: additional args to instantiate a `DataTrainingArguments`
        instance for exporting samples
    :param one_shot: one shot recipe to be applied before exporting model
    :return: path to the exported ONNX file
    """
    task = task.replace("_", "-").replace(" ", "-")

    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        raise ValueError(
            "model_path must be a directory that contains the trained transformer "
            f"files. {model_path} is not a directory or does not exist"
        )

    if num_export_samples > 0 and data_args is None:
        _LOGGER.info(
            f"--data_args is needed for exporting {num_export_samples} "
            "real samples but got None, synthetic data samples will be "
            "generated based on model input/output shapes"
        )
    data_args: Dict[str, Any] = _parse_data_args(data_args)

    _LOGGER.info(f"Attempting onnx export for model at {model_path} for task {task}")
    config_args = {"finetuning_task": finetuning_task} if finetuning_task else {}
    config = AutoConfig.from_pretrained(
        model_path,
        **config_args,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=sequence_length
    )
    model = load_task_model(task, model_path, config)
    _LOGGER.info(f"loaded model, config, and tokenizer from {model_path}")

    eval_dataset = None
    if num_export_samples > 0 and data_args:
        tokenized_dataset = load_task_dataset(
            task=task,
            tokenizer=tokenizer,
            data_args=data_args,
            model=model,
            config=config,
        )
        eval_dataset = tokenized_dataset.get("validation")
        _LOGGER.info(f"loaded validation dataset for args {data_args}")

    model = model.train()
    trainer = Trainer(
        model=model,
        model_state_path=model_path,
        eval_dataset=eval_dataset,
        recipe=None,
        recipe_args=None,
        teacher=None,
    )
    model = model.cpu()
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

    # create fake model input
    inputs = tokenizer(
        "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
    ).data  # Dict[Tensor]

    # Rearrange inputs' keys to match those defined by model foward func, which
    # seem to define how the order of inputs is determined in the exported model
    forward_args_spec = inspect.getfullargspec(model.__class__.forward)
    dropped = [
        input_key
        for input_key in inputs.keys()
        if input_key not in forward_args_spec.args
    ]
    inputs = collections.OrderedDict(
        [
            (func_input_arg_name, inputs[func_input_arg_name][0].reshape(1, -1))
            for func_input_arg_name in forward_args_spec.args
            if func_input_arg_name in inputs
        ]
    )
    if dropped:
        _LOGGER.warning(
            "The following inputs were not present in the model forward function "
            f"and therefore dropped from ONNX export: {dropped}"
        )

    inputs_shapes = {
        key: (
            f"{val.dtype if hasattr(val, 'dtype') else 'unknown'}: "
            f"{list(val.shape) if hasattr(val, 'shape') else 'unknown'}"
        )
        for key, val in inputs.items()
    }

    _LOGGER.info(f"Created sample inputs for the ONNX export process: {inputs_shapes}")

    if one_shot:
        one_shot_manager = ScheduledModifierManager.from_yaml(file_path=one_shot)
        one_shot_manager.apply(module=model)

    # run export
    model = model.eval()
    onnx_file_path = os.path.join(model_path, onnx_file_name)

    export_onnx(
        model,
        inputs,
        onnx_file_path,
        convert_qat=convert_qat,
    )
    _LOGGER.info(f"ONNX exported to {onnx_file_path}")

    # export sample inputs/outputs

    if num_export_samples > 0:
        _LOGGER.info(f"Exporting {num_export_samples} sample inputs/outputs")
        trainer.save_sample_inputs_outputs(
            num_samples_to_export=num_export_samples,
            tokenizer=tokenizer,
        )

    _LOGGER.info(f"{num_export_samples} sample inputs/outputs exported")
    return onnx_file_path


def _parse_data_args(data_args):
    try:
        return parse_recipe_variables(data_args)
    except ValueError as parse_error:
        message = str(parse_error).replace("recipe_args", "data_args")
        if "recipe variables" in message:
            message = message.replace("recipe variables", "data_args")
        raise ValueError(message)


def create_deployment_folder(
    training_directory: str,
    onnx_file_name: str = MODEL_ONNX_NAME,
    deployment_files: Optional[List[str]] = None,
):
    """
    Sets up the deployment directory i.e. copies over the complete set of files
    that are required to run the transformer model in the inference engine

    :param training_directory: path to directory where model files, tokenizers,
        and configs are saved. Exported ONNX model is also expected to be there
    :param onnx_file_name: Name for exported ONNX file in the model directory.
    :param deployment_files: optional list of deployment file names to override
        default file names with.
    :return: path to the valid deployment directory
    """

    if deployment_files is None:
        # set deployment files to default values
        deployment_files = copy.deepcopy(DEPLOYMENT_FILES)
        if onnx_file_name != MODEL_ONNX_NAME:
            # replace the default onnx model name with the custom one
            deployment_files[deployment_files.index(MODEL_ONNX_NAME)] = onnx_file_name

    model_root_dir = os.path.dirname(training_directory)
    deployment_folder_dir = os.path.join(model_root_dir, "deployment")
    if os.path.isdir(deployment_folder_dir):
        shutil.rmtree(deployment_folder_dir)
    os.makedirs(deployment_folder_dir)

    for file_name in deployment_files:
        expected_file_path = os.path.join(training_directory, file_name)
        deployment_file_path = os.path.join(deployment_folder_dir, file_name)
        if not os.path.exists(expected_file_path):
            raise ValueError(
                f"Attempting to copy {file_name} file from {expected_file_path},"
                f"but the file does not exits. Make sure that {training_directory} "
                f"contains following files: {deployment_files}"
            )
        if file_name == MODEL_ONNX_NAME:
            # moving onnx file from training to deployment directory
            shutil.move(expected_file_path, deployment_file_path)
        else:
            # copying remaining `deployment_files` from training to deployment directory
            shutil.copyfile(expected_file_path, deployment_file_path)
        _LOGGER.info(
            f"Saved {file_name} in the deployment folder at {deployment_file_path}"
        )
    return deployment_folder_dir


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
        "--no_convert_qat",
        action="store_true",
        help=("Set flag to not perform QAT to fully quantized conversion after export"),
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
        default=MODEL_ONNX_NAME,
        help=(
            "Name for exported ONNX file in the model directory. "
            "Default and recommended value for pipeline "
            f"compatibility is {MODEL_ONNX_NAME}"
        ),
    )
    parser.add_argument(
        "--num_export_samples",
        type=int,
        default=0,
        help="Number of samples (inputs/outputs) to export",
    )
    parser.add_argument(
        "--data_args",
        type=str,
        default=None,
        help="Valid json loadable args used to instantiate a `DataTrainingArguments`"
        " instance while exporting samples",
    )
    parser.add_argument(
        "--one_shot",
        type=str,
        default=None,
        help="local path or SparseZoo stub to a recipe that should be applied "
        "in a one-shot manner before exporting",
    )

    return parser.parse_args()


def export(
    task: str,
    model_path: str,
    sequence_length: int,
    no_convert_qat: bool,
    finetuning_task: str,
    onnx_file_name: str,
    num_export_samples: int = 0,
    data_args: Optional[str] = None,
    one_shot: Optional[str] = None,
):
    export_transformer_to_onnx(
        task=task,
        model_path=model_path,
        sequence_length=sequence_length,
        convert_qat=(not no_convert_qat),  # False if flagged
        finetuning_task=finetuning_task,
        onnx_file_name=onnx_file_name,
        num_export_samples=num_export_samples,
        data_args=data_args,
        one_shot=one_shot,
    )

    deployment_folder_dir = create_deployment_folder(
        training_directory=model_path, onnx_file_name=onnx_file_name
    )
    _LOGGER.info(
        f"Created deployment folder at {deployment_folder_dir} "
        f"with files: {os.listdir(deployment_folder_dir)}"
    )


def main():
    args = _parse_args()
    export(
        task=args.task,
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        no_convert_qat=args.no_convert_qat,  # False if flagged
        finetuning_task=args.finetuning_task,
        onnx_file_name=args.onnx_file_name,
        num_export_samples=args.num_export_samples,
        data_args=args.data_args,
        one_shot=args.one_shot,
    )


if __name__ == "__main__":
    main()
