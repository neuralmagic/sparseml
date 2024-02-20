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

import csv
import json
import os
import tempfile
from io import StringIO
from pathlib import Path

import pytest
import torch

from sparseml.transformers import (
    SparseAutoModelForCausalLM,
    SparseAutoTokenizer,
    apply,
    compress,
    load_dataset,
    oneshot,
    train,
)


def test_oneshot_and_finetune(tmp_path: Path):
    recipe_str = "tests/sparseml/transformers/finetune/test_alternate_recipe.yaml"
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    concatenate_data = True
    run_stages = True
    output_dir = tmp_path
    max_steps = 50
    splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

    apply(
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
    recipe_str = "tests/sparseml/transformers/obcq/test_tiny2.yaml"
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

    recipe_str = "tests/sparseml/transformers/finetune/test_finetune_recipe.yaml"
    model = tmp_path / "oneshot_out"
    dataset = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path / "finetune_out"
    splits = "train[:50%]"
    max_steps = 50

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


def test_finetune_wout_recipe(tmp_path: Path):
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


@pytest.mark.parametrize("file_extension", ["json", "csv"])
def test_finetune_wout_recipe_custom_dataset(
    file_extension,
    tmp_path: Path,
):
    recipe_str = None
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    dataset_path = Path(tempfile.mkdtemp())

    created_success = _create_mock_custom_dataset_folder_structure(
        dataset_path, file_extension
    )
    assert created_success

    def preprocessing_func(example):
        example["text"] = "Review: " + example["text"]
        return example

    concatenate_data = False
    output_dir = tmp_path
    max_steps = 50
    train(
        model=model,
        dataset=file_extension,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        oneshot_device=device,
        text_column="text",
        dataset_path=dataset_path,
        preprocessing_func=preprocessing_func,
    )


def _create_mock_custom_dataset_folder_structure(tmp_dir_data, file_extension):
    train_path = os.path.join(tmp_dir_data, "train")
    test_path = os.path.join(tmp_dir_data, "test")
    validate_path = os.path.join(tmp_dir_data, "validate")

    # create tmp mock data files
    create_mock_file(
        extension=file_extension,
        content="text for train data 1",
        path=train_path,
        filename="data1",
    )
    create_mock_file(
        extension=file_extension,
        content="text for train data 2",
        path=train_path,
        filename="data2",
    )
    create_mock_file(
        extension=file_extension,
        content="text for test data 1",
        path=test_path,
        filename="data3",
    )
    create_mock_file(
        extension=file_extension,
        content="text for validate data 1",
        path=validate_path,
        filename="data4",
    )
    return True


def create_mock_file(extension, content, path, filename):

    os.makedirs(path, exist_ok=True)

    if extension == "json":
        mock_data = {"text": content}
        mock_content = json.dumps(mock_data, indent=2)

    else:
        fieldnames = ["text"]
        mock_data = [{"text": content}]
        csv_output = StringIO()
        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(mock_data)
        mock_content = csv_output.getvalue()

    mock_filename = f"{filename}.{extension}"
    mock_filepath = os.path.join(path, mock_filename)

    with open(mock_filepath, "w") as mock_file:
        mock_file.write(mock_content)

    return mock_filepath  # Return the file path
