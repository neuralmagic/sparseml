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

from transformers import OPTForCausalLM

from sparseml.pytorch.sparsification.obcq.data import get_c4, get_ptb, get_wikitext2

# from sparseml.pytorch.sparsification.obcq.sparse_opt_modifier import SparseOPTModifier
from sparseml.pytorch.sparsification.obcq.manager import RecipeManagerOneShot


__all__ = ["one_shot"]

_LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = ["wikitext2", "ptb", "c4"]


def one_shot(
    model_path: str,
    dataset_name: str,
    deploy_dir: str = ".",
    num_samples: int = 128,
    recipe_file: Optional[str] = None,
) -> None:
    """
    Performs in place one shot sparsification/quantization of a model based on:

    :param model_path: path to Hugging Face stub
    :param dataset_name: Dataset to extract calibration data from
    :param deploy_dir: The output directory to save the model to
    :param num_samples: Number of samples to extract from the dataset
    :param recipe_file: recipe containing SparseGPT configuration
    """
    deploy_dir = Path(os.path.join(deploy_dir, "deployment"))

    if deploy_dir.exists():
        raise RuntimeError(f"deploy_dir={deploy_dir} already exists")

    # TODO: don't hardcode this for OPT
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings

    data_loader_fn = None
    if dataset_name == "wikitext2":
        data_loader_fn = get_wikitext2
    elif dataset_name == "ptb":
        data_loader_fn = get_ptb
    elif dataset_name == "c4":
        data_loader_fn = get_c4
    else:
        raise ValueError(
            f"dataset_name={dataset_name} should be one of {SUPPORTED_DATASETS}"
        )

    calibration_data, test_encoder, tokenizer = data_loader_fn(
        num_samples, 0, model.seqlen, model_path
    )

    recipe = RecipeManagerOneShot.from_yaml(recipe_file)
    """
    sparse_opt_mod = SparseOPTModifier(
        sparsity=0.5,
        block_size=128,
        quantize=True,
        num_bits=8
    )
    recipe = RecipeManagerOneShot([sparse_opt_mod])
    """
    recipe.one_shot(model, calibration_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="OPT model to load; pass `facebook/opt-X`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument("--deploy-dir", type=str, default=".")
    parser.add_argument("--recipe", type=str, default=None)

    args = parser.parse_args()

    one_shot(
        model_path=args.model,
        dataset_name=args.dataset,
        deploy_dir=args.deploy_dir,
        num_samples=args.nsamples,
        recipe_file=args.recipe,
    )
