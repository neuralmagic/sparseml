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
from typing import Optional

from torch.nn import Module
from transformers import AutoConfig

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.modifiers.obcq.utils.helpers import ppl_eval_general
from sparseml.pytorch.model_load.helpers import (
    RECIPE_FILE_NAME,
    apply_recipe_structure_to_model,
    fallback_to_cpu,
    parse_dtype,
    save_model_and_recipe,
)
from sparseml.transformers.data import TransformersDataset
from sparseml.transformers.sparsification.obcq.utils.helpers import (
    llama_forward,
    opt_forward,
)
from sparseml.transformers.utils.model import SparseCausalLM


__all__ = ["one_shot"]

_LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = TransformersDataset.registered_names()
SUPPORTED_MODELS = ["opt", "llama", "mistral"]
SUPPORTED_PRECISION = ["auto", "half", "full", "float16", "bfloat16", "float32"]


def one_shot(
    model_path: str,
    dataset_name: str,
    num_samples: int = 128,
    sequence_length: Optional[int] = None,
    device: str = "cuda:0",
    deploy_dir: Optional[str] = ".",
    recipe_file: Optional[str] = None,
    precision: str = "auto",
    eval_data: Optional[str] = None,
    do_save: Optional[bool] = False,
) -> Module:
    """
    Performs in place one shot sparsification/quantization of a model based on:

    :param model_path: path to Hugging Face stub
    :param dataset_name: Dataset to extract calibration data from
    :param num_samples: Number of samples to extract from the dataset
    :param sequence_length: Maximum input sequence length to the model
    :param device: Device (cuda:index or cpu) to use for computation
    :param deploy_dir: The output directory to save the model to
    :param recipe_file: recipe containing SparseGPT configuration
    :param precision: precision to load model as, either auto, half or full
    :param eval_data: dataset to use for perplexity evalaution, or none to skip
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
    model_type = config.model_type.lower()

    model_loader_fn = None
    forward_fn = None
    if "opt" in model_type:
        model_loader_fn = SparseCausalLM.opt_model_from_pretrained
        forward_fn = opt_forward
    elif "llama" in model_type or "mistral" in model_type:
        model_loader_fn = SparseCausalLM.auto_model_from_pretrained
        forward_fn = llama_forward
    else:
        _LOGGER.warning(
            f"A supported model type({SUPPORTED_MODELS}) could not be "
            f"parsed from model_path={model_path}. Defaulting to "
            "AutoModelForCausalLM loading. "
        )
        model_loader_fn = SparseCausalLM.auto_model_from_pretrained
        forward_fn = llama_forward
    torch_dtype = parse_dtype(precision)
    model = model_loader_fn(
        model_path, sequence_length=sequence_length, torch_dtype=torch_dtype
    )

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset_name={dataset_name} should be one of {SUPPORTED_DATASETS}"
        )
    dataset = TransformersDataset.load_from_registry(
        dataset_name,
        model=model_path,
        seqlen=sequence_length or model.seqlen,
        nsamples=num_samples,
        seed=0,
        split="train",
    )
    calibration_data = dataset.loader
    tokenizer = dataset.tokenizer

    # create session and initialize any structure from input model recipe
    session_manager.create_session()
    session = session_manager.active_session()
    input_recipe_path = os.path.join(model_path, RECIPE_FILE_NAME)
    if os.path.exists(input_recipe_path):
        apply_recipe_structure_to_model(
            model=model, recipe_path=input_recipe_path, model_path=model_path
        )

    # launch one shot
    session.apply(
        framework=Framework.pytorch,
        recipe=recipe_file,
        model=model,
        calib_data=calibration_data,
        start=-1,
        device=device,
        copy_data=False,
    )

    if do_save:
        save_model_and_recipe(model, deploy_dir, tokenizer)
    if eval_data:
        dataset = TransformersDataset.load_from_registry(
            eval_data,
            model=model_path,
            seqlen=model.seqlen,
            nsamples=None,
            seed=0,
            split="test",
            split_percent_to_use=0.1 if eval_data == "open_platypus" else 1.0,
        )
        test_data = dataset.loader
        ppl_eval_general(
            forward_fn, model, test_data, device, max_samples_per_iteration=8
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Hugging Face stub of model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Name of dataset to extract calibration data from",
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
        "--eval", type=str, default=None, help="Optional dataset for perplexity eval"
    )
    parser.add_argument(
        "--save", type=bool, default=False, help="Save output model to disk"
    )

    args = parser.parse_args()

    one_shot(
        model_path=args.model,
        dataset_name=args.dataset,
        deploy_dir=args.deploy_dir,
        num_samples=args.nsamples,
        sequence_length=args.seqlen,
        device=args.device,
        recipe_file=args.recipe,
        precision=args.precision,
        eval_data=args.eval,
        do_save=args.save,
    )
