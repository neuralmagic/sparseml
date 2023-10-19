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

import sparseml.core.session as sml
from sparseml.core.framework import Framework
from sparseml.modifiers.obcq.utils.helpers import ppl_eval_general
from sparseml.optim.helpers import load_recipe_yaml_str
from sparseml.transformers.data import TransformersDataset
from sparseml.transformers.utils.model import SparseCasualLM
from sparseml.transformers.sparsification.trainer import Trainer

_LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = ["wikitext2", "ptb", "c4", "open_platypus"]
SUPPORTED_MODELS = ["opt", "llama"]


def main(
    model_path: str,
    dataset_name: str,
    num_samples: int = 128,
    device: str = "cuda:0",
    deploy_dir: Optional[str] = ".",
    recipe_file: Optional[str] = None,
    do_save: Optional[bool] = False,
) -> Module:
    if do_save:
        deploy_dir = Path(os.path.join(deploy_dir, "test_deployment"))

        if deploy_dir.exists():
            raise RuntimeError(f"deploy_dir={deploy_dir} already exists")

    model_loader_fn = None
    if "opt" in model_path.lower():
        model_loader_fn = SparseCasualLM.opt_model_from_pretrained
    elif "llama" in model_path.lower():
        model_loader_fn = SparseCasualLM.llama_model_from_pretrained
    else:
        raise ValueError(f"model_path={model_path} should be one of {SUPPORTED_MODELS}")
    model = model_loader_fn(model_path)

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset_name={dataset_name} should be one of {SUPPORTED_DATASETS}"
        )
    
    from datasets import load_dataset
    raw_dataset = load_dataset(dataset_name)

    dataset = TransformersDataset.load_from_registry(
        dataset_name,
        model=model_path,
        seqlen=model.seqlen,
        nsamples=num_samples,
        seed=0,
        split="train",
    )
    train_data = dataset.loader
    tokenizer = dataset.tokenizer

    trainer = Trainer(
        model=model,
        model_state_path=model_path,
        recipe=recipe_file,
        train_dataset = train_data,
        tokenizer=tokenizer
    )
    train_result = trainer.train()

    if do_save:
        _save(model, tokenizer, deploy_dir, recipe_file)

    return model


def _save(model, tokenizer, save_path, recipe_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    _LOGGER.info("Saving output to {}".format(os.path.abspath(save_path)))
    recipe_output_path = os.path.join(save_path, "recipe.yaml")
    with open(recipe_output_path, "w") as fp:
        fp.write(load_recipe_yaml_str(recipe_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Hugging Face stub of model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "open_platypus"],
        help="Name of dataset to extract calibration data from",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--deploy-dir", type=str, default=".")
    parser.add_argument("--recipe", type=str, default=None)
    parser.add_argument(
        "--save", type=bool, default=False, help="Save output model to disk"
    )

    args = parser.parse_args()

    main(
        model_path=args.model,
        dataset_name=args.dataset,
        deploy_dir=args.deploy_dir,
        num_samples=args.nsamples,
        device=args.device,
        recipe_file=args.recipe,
        do_save=args.save,
    )