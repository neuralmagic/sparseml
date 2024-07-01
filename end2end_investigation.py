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
To run without FSDP: python this_script.py {ARGS}

To run with FSDP: accelerate launch
--config_file integrations/huggingface-transformers/finetuning/example_fsdp_config.yaml python this_script.py --fsdp {ARGS}
(by tweaking `num_processes` in the config file, you can control the number of parallel processes running in the context of FSDP)
"""

import argparse

import torch

from sparseml.transformers import SparseAutoModelForCausalLM, apply, oneshot, train


recipe_pruning = """
sparsity_stage:
  run_type: oneshot
  sparsity_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      mask_structure: "2:4"
      sequential_update: false"""

recipe_finetuning = """
finetuning_stage:
  run_type: train
  finetuning_modifiers:
    ConstantPruningModifier:
      targets: [
        're:.*q_proj.weight',
        're:.*k_proj.weight',
        're:.*v_proj.weight',
        're:.*o_proj.weight',
        're:.*gate_proj.weight',
        're:.*up_proj.weight',
        're:.*down_proj.weight',
      ]
      start: 0"""

recipe_quantization = """
quantization_stage:
  run_type: oneshot
  quantization_modifiers:
    GPTQModifier:
      sequential_update: false
      config_groups:
        group_0:
          weights:
            num_bits: 4
            type: "int"
            symmetric: true
            strategy: "channel"
          targets: ["Linear"]"""

training_hyperparams = dict(
    bf16=False,  # use full precision for training
    max_seq_length=512,
    num_train_epochs=0.5,
    logging_steps=500,
    save_steps=5000,
    gradient_checkpointing=True,
    learning_rate=0.0001,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)


def run_one_shot(
    model_stub: str,
    recipe: str,
    fsdp: bool,
    target: str,
    dataset: str,
    split_name: str,
    num_samples: int,
):
    # apply oneshot pruning or quantization to the model
    if not fsdp:
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_stub, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        # device_map not supported with accelerate
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_stub, torch_dtype=torch.bfloat16
        )

    output_dir = f"{target}_{model_stub.split('/')[-1]}"  # e.g. pruning_Xenova/llama2.c-stories15M

    # needs to be adjusted wrt to the dataset
    splits = {"calibration": f"{split_name}[:5%]"}
    num_calibration_samples = num_samples

    print(
        f"Running {target} on {model_stub} with recipe: {recipe}\nfsdp: {fsdp}, dataset: {dataset}, split_name: {split_name}, num_samples: {num_samples}"
    )

    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        splits=splits,
    )


def run_finetuning(
    model_stub: str,
    recipe: str,
    fsdp: bool,
    target: str,
    dataset: str,
    split_name: str,
    training_hyperparams=training_hyperparams,
):
    # apply finetuning to the model
    if not fsdp:
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_stub, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        # device_map not supported with accelerate
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_stub, torch_dtype=torch.bfloat16
        )

    output_dir = f"{target}_{model_stub.split('/')[-1]}"

    splits = {"train": f"{split_name}[:5%]"}

    print(
        f"Running {target} on {model_stub} with recipe: {recipe}\nfsdp: {fsdp}, dataset: {dataset}, split_name: {split_name}"
    )

    train(
        model=model,
        dataset=dataset,
        recipe=recipe,
        output_dir=output_dir,
        splits=splits,
        **training_hyperparams,
    )


def run_end2end(
    model_stub: str,
    recipe: str,
    fsdp: bool,
    target: str,
    dataset: str,
    split_name_calib: str,
    split_name_train: str,
    num_samples: int,
):
    # run compression end-to-end (recipe is a sum of several stages)
    if not fsdp:
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_stub, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        # device_map not supported with accelerate
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_stub, torch_dtype=torch.bfloat16
        )

    output_dir = f"{target}_{model_stub.split('/')[-1]}"

    splits = {
        "train": f"{split_name_train}[:5%]",
        "calibration": f"{split_name_calib}[:5%]",
    }

    print(
        f"Running {target} on {model_stub} with recipe: {recipe}\nfsdp: {fsdp}, dataset: {dataset}, "
        f"split_names: {split_name_train} (train) {split_name_calib} (calib), num_samples: {num_samples}"
    )

    apply(
        model=model,
        dataset=dataset,
        recipe=recipe,
        output_dir=output_dir,
        num_calibration_samples=num_samples,
        splits=splits,
        **training_hyperparams,
    )


def run(args):
    if args.target == "pruning":
        run_one_shot(
            model_stub=args.model,
            recipe=recipe_pruning,
            fsdp=args.fsdp,
            target=args.target,
            dataset=args.dataset,
            split_name=args.split_name_calib,
            num_samples=args.num_samples,
        )
    elif args.target == "finetuning":
        run_finetuning(
            model_stub=args.model,
            recipe=recipe_pruning,
            fsdp=args.fsdp,
            target=args.target,
            dataset=args.dataset,
            split_name=args.split_name_train,
        )
    elif args.target == "quantization":
        run_one_shot(
            model_stub=args.model,
            recipe=recipe_quantization,
            fsdp=args.fsdp,
            target=args.target,
            dataset=args.dataset,
            split_name=args.split_name_calib,
            num_samples=args.num_samples,
        )
    elif args.target == "end2end":
        run_end2end(
            model_stub=args.model,
            recipe=recipe_pruning + recipe_finetuning + recipe_quantization,
            fsdp=args.fsdp,
            target=args.target,
            dataset=args.dataset,
            split_name_calib=args.split_name_calib,
            split_name_train=args.split_name_train,
            num_samples=args.num_samples,
        )
    else:
        raise ValueError("Invalid target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Xenova/llama2.c-stories15M",
        help="Model to be tested",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["pruning", "finetuning", "quantization", "end2end"],
        default="pruning",
        help="Compression mode to be tested",
    )
    parser.add_argument("--fsdp", action="store_true", help="Are we running with FSDP?")
    parser.add_argument(
        "--dataset", type=str, default="open-platypus", help="Dataset name"
    )
    parser.add_argument(
        "--split_name_calib",
        type=str,
        default="train",
        help="Name of the split used for calibration",
    )
    parser.add_argument(
        "--split_name_train",
        type=str,
        default="train",
        help="Name of the split used for finetuning",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=512,
        help="Number of samples used for calibration",
    )
    args = parser.parse_args()
    run(args)
