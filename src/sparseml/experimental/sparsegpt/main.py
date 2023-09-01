import os
import time

import torch

from sparseml.optim.helpers import load_recipe_yaml_str
from sparseml.experimental.sparsegpt.dispatch import (
    load_data,
    load_model,
    prepare_sparsegpt,
)

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


DEV = torch.device("cuda:0")


@torch.no_grad()
def sequential(model, dataloader, dev, args):
    sequential_sparsegpt = prepare_sparsegpt(model, dataloader, args)
    sequential_sparsegpt.compress(dev)


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
        "model", type=str, help="OPT model to load; pass `facebook/opt-X`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "open_platypus", "platypus"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--recipe", type=str, default=None)
    parser.add_argument("--observer-batches", type=int, default=100)
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="Whether to perform per-channel quantization.",
    )
    parser.add_argument(
        "--smoothquant", type=int, default=0, help="Whether to run SmoothQuant."
    )
    parser.add_argument(
        "--smooth-activation-file",
        type=str,
        help="Activation file to be used with SmoothQuant",
        default=None,
    )
    parser.add_argument("--ptq", type=int, default=0, help="Whether to run PTQ.")
    parser.add_argument(
        "--ptq-init",
        type=int,
        default=0,
        help="Whether to initialize quantization parameters using PTQ",
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
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
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
        "--eval-dense", type=int, default=0, help="Whether to evaluate dense model."
    )

    # For MPT
    parser.add_argument(
        "--yaml-path", type=str, default="", help="Path to recipe yaml."
    )
    parser.add_argument("--args-list", type=str, default="", help="Args list.")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model, seqlen = load_model(args)
    dataloader, testloader, tokenizer = load_data(args, seqlen)

    if args.wbits < 16 or ((args.sparsity or args.prunen) and not args.gmp):
        tick = time.time()
        sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    if args.save:
        _save(model, tokenizer, args.save)
