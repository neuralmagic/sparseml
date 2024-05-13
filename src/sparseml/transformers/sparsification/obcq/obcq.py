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

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from torch.nn import Module
from transformers import AutoConfig

import sparseml
from sparseml.core.framework import Framework
from sparseml.pytorch.model_load.helpers import (
    fallback_to_cpu,
    parse_dtype,
    save_model_and_recipe,
)
from sparseml.transformers import SparseAutoTokenizer
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import format_calibration_data
from sparseml.transformers.utils.helpers import resolve_sequence_length
from sparseml.transformers.utils.initializers import initialize_sparse_model


__all__ = ["one_shot"]

_LOGGER = logging.getLogger(__name__)
SUPPORTED_MODELS = ["opt", "llama", "mistral"]
SUPPORTED_PRECISION = ["auto", "half", "full", "float16", "bfloat16", "float32"]


def one_shot(
    model_path: str,
    dataset: str,
    dataset_config_name: Optional[str] = None,
    num_samples: int = 128,
    sequence_length: Optional[int] = None,
    concatenate_data: Optional[bool] = False,
    device: str = "cuda:0",
    deploy_dir: Optional[str] = ".",
    recipe_file: Optional[str] = None,
    precision: str = "auto",
    recipe_args: Optional[Dict] = None,
    do_save: Optional[bool] = False,
) -> Module:
    """
    Performs in place one shot sparsification/quantization of a model based on:

    :param model_path: path to Hugging Face stub
    :param dataset: Dataset to extract calibration data from
    :param dataset_config_name: Specific configuration to extract from calib dataset
    :param num_samples: Number of samples to extract from the dataset
    :param sequence_length: Maximum input sequence length to the model
    :param concatenate_data: Whether to concatenate datapoints to fill seqlen or not
    :param device: Device (cuda:index, auto or cpu) to use for computation
    :param deploy_dir: The output directory to save the model to
    :param recipe_file: recipe containing SparseGPT configuration
    :param precision: precision to load model as, either auto, half or full
    :param recipe_args: additional arguments to use for recipe evaluation
    :param do_save: whether to save the output model to disk

    :return: Pytorch module with OBCQ applied
    """

    if do_save:
        deploy_dir = Path(os.path.join(deploy_dir, "obcq_deployment"))

        if deploy_dir.exists():
            raise RuntimeError(f"deploy_dir={deploy_dir} already exists")

    # fallback to cpu if cuda not available
    device = fallback_to_cpu(device)
    _LOGGER.info(f"Running one_shot on device {device}")

    # Load the configuration from the model path
    config = AutoConfig.from_pretrained(model_path)

    torch_dtype = parse_dtype(precision)
    sparseml.create_session()
    model = initialize_sparse_model(
        model_path=model_path,
        task="text-generation",
        sequence_length=sequence_length,
        torch_dtype=torch_dtype,
        config=config,
        device_map=device,
    )

    # Load calibration data
    try:
        TextGenerationDataset.get_value_from_registry(dataset)
    except KeyError:
        raise ValueError(
            f"dataset={dataset} should be one of "
            f"{TextGenerationDataset.registered_names()}"
        )

    data_args = DataTrainingArguments(
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        max_seq_length=sequence_length or resolve_sequence_length(config),
        num_calibration_samples=num_samples,
        concatenate_data=concatenate_data,
        pad_to_max_length=False,
    )

    tokenizer = SparseAutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    dataset_manager = TextGenerationDataset.load_from_registry(
        dataset, data_args=data_args, split="train", tokenizer=tokenizer
    )
    raw_dataset = dataset_manager.get_raw_dataset()
    tokenized_dataset = dataset_manager.tokenize_and_process(raw_dataset)
    calibration_data = format_calibration_data(
        tokenized_dataset=tokenized_dataset, num_calibration_samples=num_samples
    )

    # launch one shot
    session = sparseml.active_session()
    session.apply(
        framework=Framework.pytorch,
        recipe=recipe_file,
        model=model,
        calib_data=calibration_data,
        start=-1,
        copy_data=False,
        recipe_args=recipe_args,
    )

    if do_save:
        save_model_and_recipe(model, deploy_dir, tokenizer)

    return model


class KeyValue(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Hugging Face stub of model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=TextGenerationDataset.registered_names(),
        help="Name of dataset to extract calibration data from",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Specific configuration to extract from calibration dataset (optional)",
    )
    parser.add_argument(
        "--nsamples", type=int, default=512, help="Number of calibration data samples"
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Maximum input sequence length to the model",
    )
    parser.add_argument(
        "--concat_data",
        type=bool,
        default=False,
        help="Whether or not to concatenate samples to fill sequence length",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--deploy-dir", type=str, default=".")
    parser.add_argument("--recipe", type=str, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        choices=SUPPORTED_PRECISION,
        default="auto",
        help="Precision to cast model weights to, default to auto",
    )
    parser.add_argument(
        "--recipe_args",
        nargs="*",
        action=KeyValue,
        help="Recipe arguments to evaluate, of the format key1=value1 key2=value2",
    )
    parser.add_argument(
        "--save", type=bool, default=True, help="Save output model to disk"
    )

    args = parser.parse_args()

    one_shot(
        model_path=args.model,
        dataset=args.dataset,
        dataset_config_name=args.dataset_config,
        deploy_dir=args.deploy_dir,
        num_samples=args.nsamples,
        sequence_length=args.seqlen,
        concatenate_data=args.concat_data,
        device=args.device,
        recipe_file=args.recipe,
        precision=args.precision,
        recipe_args=args.recipe_args,
        do_save=args.save,
    )
