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

import os
import time

import torch

from sparseml.experimental.sparsegpt.dispatch import (
    evaluate_perplexity,
    load_data,
    load_model,
    prepare_sparsegpt,
)
from sparseml.optim.helpers import load_recipe_yaml_str


try:
    import wandb

    has_wandb = True
except Exception:
    has_wandb = False


@torch.no_grad()
def sequential(model, dataloader, dev, args):
    sequential_sparsegpt = prepare_sparsegpt(model, dataloader, args=args, dev=dev)
    if args.ptq_only:
        sequential_sparsegpt.pre_compress(dev=dev)
    else:
        sequential_sparsegpt.compress(dataloader=dataloader, dev=dev)


def _save(model, tokenizer, save_path):
    assert save_path, "Save path must be speficied"
    print(f"Saving model and artifacts to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if args.recipe:
        recipe_path = os.path.join(save_path, "recipe.yaml")
        with open(recipe_path, "w") as fp:
            fp.write(load_recipe_yaml_str(args.recipe))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="Model to load; e.g., `facebook/opt-1.3b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "open_platypus", "platypus"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--data-sequence-length", type=int, default=2048)
    parser.add_argument("--recipe", type=str, default=None)
    parser.add_argument("--observer-batches", type=int, default=100)
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="Whether to perform per-channel quantization.",
    )
    parser.add_argument(
        "--smoothquant", action="store_true", help="Whether to run SmoothQuant."
    )
    parser.add_argument(
        "--smooth-activation-file",
        type=str,
        help="Activation file to be used with SmoothQuant",
        default=None,
    )
    parser.add_argument(
        "--ptq-only", action="store_true", help="Flag to perform only PTQ step."
    )
    parser.add_argument(
        "--ptq-init",
        type=int,
        default=0,
        help="Whether to initialize quantization parameters using PTQ",
    )
    parser.add_argument(
        "--sequential_hessian_within_layer",
        type=int,
        default=0,
        help="Whether to compute Hessian for modules with sequential "
        "compression within layer",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--save-ptq", type=str, default="", help="Path to saved PTQ model."
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--eval", action="store_true", help="Whether to evaluate perplexity at the end."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Whether to evaluate perplexity at the end.",
    )

    # For MPT
    parser.add_argument(
        "--yaml-path", type=str, default="", help="Path to recipe yaml."
    )
    parser.add_argument("--args-list", type=str, default="", help="Args list.")

    args = parser.parse_args()
    DEV = torch.device(args.device)

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    print("Load model", flush=True)
    model, seqlen = load_model(args)

    print("Load data", flush=True)
    dataloader, testloader, tokenizer = load_data(args, None, seqlen)

    tick = time.time()
    sequential(model, dataloader, DEV, args)
    print(time.time() - tick)

    if args.save:
        _save(model, tokenizer, args.save)

    if args.eval:
        _, testloader, _ = load_data(args, dataset="wikitext2")
        evaluate_perplexity(model, testloader, DEV)
