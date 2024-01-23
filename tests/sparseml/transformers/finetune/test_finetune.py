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

from pathlib import Path
import torch

import sparseml.core.session as session_manager
from sparseml.transformers.finetune.text_generation import (
    run_general,
    run_oneshot,
    run_train,
)


def test_oneshot_and_finetune(tmp_path: Path):
    recipe_str = "tests/sparseml/transformers/finetune/test_alternate_recipe.yaml"
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    concatenate_data = True
    run_stages = True
    output_dir = tmp_path
    max_steps = 50
    splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

    run_general(
        model_name_or_path=model,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        run_stages=run_stages,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )


def test_oneshot_then_finetune(tmp_path: Path):
    recipe_str = "tests/sparseml/transformers/obcq/test_tiny2.yaml"
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset_name = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 64
    output_dir = tmp_path / "oneshot_out"
    splits = {"calibration": "train[:10%]"}

    run_oneshot(
        model_name_or_path=model,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )

    session = session_manager.active_session()
    session.reset()

    recipe_str = "tests/sparseml/transformers/finetune/test_finetune_recipe.yaml"
    model = tmp_path / "oneshot_out"
    dataset_name = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path / "finetune_out"
    splits = "train[:50%]"
    max_steps = 50

    run_train(
        model_name_or_path=model,
        distill_teacher="Xenova/llama2.c-stories15M",
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        max_steps=max_steps,
    )


def test_finetune_wout_recipe(tmp_path: Path):
    recipe_str = None
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset_name = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path
    max_steps = 50
    splits = "train"
    run_train(
        model_name_or_path=model,
        dataset_name=dataset_name,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )
    
def test_finetune_wout_recipe_custom_dataset(tmp_path: Path):
    recipe_str = None
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset_name = "json" or "csv"
    path = "tests/sparseml/transformers/finetune/data/train/"

    data_files ={
        "train": [
            path + "data1.json",
            path + 'data2.json',
        ],
        "test": [
            path + "data1.json",
            path + 'data2.json',
        ]
}

    raw_kwargs = {"data_files": data_files}
    # load_dataset("json", data_files="my_file.json")
    concatenate_data = False
    output_dir = tmp_path
    max_steps = 50
    splits = "train"
    run_train(
        model_name_or_path=model,
        dataset_name=dataset_name,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
        raw_kwargs=raw_kwargs,
    )