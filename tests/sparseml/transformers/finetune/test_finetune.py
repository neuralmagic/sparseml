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
from pathlib import Path

import pytest
import torch

import sparseml
from sparseml.modifiers.obcq.base import SparseGPTModifier
from sparseml.transformers import (
    SparseAutoModelForCausalLM,
    SparseAutoTokenizer,
    compress,
    load_dataset,
    oneshot,
    train,
)

def test_oneshot_and_finetune_with_tokenizer(tmp_path: Path):
    recipe_str = "tests/sparseml/transformers/finetune/test_alternate_recipe.yaml"
    model = SparseAutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")
    tokenizer = SparseAutoTokenizer.from_pretrained(
        "Xenova/llama2.c-stories15M",
    )
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    dataset_config_name = "wikitext-2-raw-v1"
    dataset = load_dataset("wikitext", dataset_config_name, split="train[:50%]")
    # dataset ="wikitext"

    concatenate_data = True
    run_stages = True
    output_dir = tmp_path
    max_steps = 50
    splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

    compress(
        model=model,
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        run_stages=run_stages,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        tokenizer=tokenizer,
    )


def test_oneshot_then_finetune(tmp_path: Path):
    recipe_str = "tests/sparseml/transformers/obcq/recipes/test_tiny2.yaml"
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    output_dir = tmp_path / "oneshot_out"
    splits = {"calibration": "train[:10%]"}

    with sparseml.create_session():
        oneshot(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe_str,
            concatenate_data=concatenate_data,
            splits=splits,
            oneshot_device=device,
        )

    recipe_str = "tests/sparseml/transformers/finetune/test_finetune_recipe.yaml"
    model = tmp_path / "oneshot_out"
    dataset = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path / "finetune_out"
    splits = "train[:50%]"
    max_steps = 50

    with sparseml.create_session():
        train(
            model=model,
            distill_teacher="Xenova/llama2.c-stories15M",
            dataset=dataset,
            output_dir=output_dir,
            num_calibration_samples=num_calibration_samples,
            recipe=recipe_str,
            concatenate_data=concatenate_data,
            splits=splits,
            max_steps=max_steps,
        )


def test_finetune_without_recipe(tmp_path: Path):
    recipe_str = None
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path
    max_steps = 50
    splits = "train"

    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )

def test_safetensors(tmp_path: Path):
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    output_dir = tmp_path / "output1"
    max_steps = 10
    splits = {"train": "train[:10%]"}

    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        max_steps=max_steps,
        splits=splits,
        oneshot_device=device,
    )

    assert os.path.exists(output_dir / "model.safetensors")
    assert not os.path.exists(output_dir / "pytorch_model.bin")

    # test we can also load
    new_output_dir = tmp_path / "output2"
    train(
        model=output_dir,
        dataset=dataset,
        output_dir=new_output_dir,
        max_steps=max_steps,
        splits=splits,
        oneshot_device=device,
    )


def test_oneshot_with_modifier_object(tmp_path: Path):
    recipe_str = [SparseGPTModifier(sparsity=0.5, targets=[r"re:model.layers.\d+$"])]
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    output_dir = tmp_path / "oneshot_out"
    splits = {"calibration": "train[:10%]"}

    oneshot(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )
