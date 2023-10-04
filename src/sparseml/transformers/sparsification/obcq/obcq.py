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
from sparseml.modifiers.obcq.utils.data import (
    get_c4,
    get_openplatypus,
    get_ptb,
    get_wikitext2,
)
from sparseml.modifiers.obcq.utils.models import (
    llama_forward,
    load_llama_model,
    load_opt_model,
    opt_forward,
)
from sparseml.modifiers.obcq.utils.utils import ppl_eval_general
from sparseml.optim.helpers import load_recipe_yaml_str


__all__ = ["one_shot"]

_LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = ["wikitext2", "ptb", "c4", "open_platypus"]
SUPPORTED_MODELS = ["opt", "llama"]


def one_shot(
    model_path: str,
    dataset_name: str,
    num_samples: int = 128,
    device: str = "cuda:0",
    deploy_dir: Optional[str] = ".",
    recipe_file: Optional[str] = None,
    do_eval: Optional[bool] = False,
    do_save: Optional[bool] = False,
) -> Module:
    """
    Performs in place one shot sparsification/quantization of a model based on:

    :param model_path: path to Hugging Face stub
    :param dataset_name: Dataset to extract calibration data from
    :param num_samples: Number of samples to extract from the dataset
    :param device: Device (cuda:index or cpu) to use for computation
    :param deploy_dir: The output directory to save the model to
    :param recipe_file: recipe containing SparseGPT configuration
    :param do_eval: whether to run perplexity evaluation on output model
    :param do_save: whether to save the output model to disk

    :return: Pytorch module with OBCQ applied
    """

    if do_save:
        deploy_dir = Path(os.path.join(deploy_dir, "obcq_deployment"))

        if deploy_dir.exists():
            raise RuntimeError(f"deploy_dir={deploy_dir} already exists")

    model_loader_fn = None
    forward_fn = None
    if "opt" in model_path.lower():
        model_loader_fn = load_opt_model
        forward_fn = opt_forward
    elif "llama" in model_path.lower():
        model_loader_fn = load_llama_model
        forward_fn = llama_forward
    else:
        raise ValueError(f"model_path={model_path} should be one of {SUPPORTED_MODELS}")
    model = model_loader_fn(model_path)

    data_loader_fn = None
    if dataset_name == "wikitext2":
        data_loader_fn = get_wikitext2
    elif dataset_name == "ptb":
        data_loader_fn = get_ptb
    elif dataset_name == "c4":
        data_loader_fn = get_c4
    elif dataset_name == "open_platypus":
        data_loader_fn = get_openplatypus
    else:
        raise ValueError(
            f"dataset_name={dataset_name} should be one of {SUPPORTED_DATASETS}"
        )

    calibration_data, _, tokenizer = data_loader_fn(
        num_samples, 0, model.seqlen, model_path
    )

    sml.create_session()
    session = sml.active_session()
    session.apply(
        framework=Framework.pytorch,
        recipe=recipe_file,
        model=model,
        calib_data=calibration_data,
        start=0.0,
        device=device,
        copy_data=False,
    )

    if do_save:
        _save(model, tokenizer, deploy_dir, recipe_file)
    if do_eval:
        _, test_dataloader, _ = get_openplatypus(
            num_samples, 0, model.seqlen, model_path
        )
        ppl_eval_general(forward_fn, model, test_dataloader, device, None, num_samples)

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
        "--eval", type=bool, default=False, help="Run perplexity evaluation"
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
        device=args.device,
        recipe_file=args.recipe,
        do_eval=args.eval,
        do_save=args.save,
    )
