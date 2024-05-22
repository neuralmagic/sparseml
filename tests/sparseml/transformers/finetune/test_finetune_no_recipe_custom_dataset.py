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
import shutil
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import pytest

from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_gpu, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/finetune/finetune_custom"
GPU_CONFIGS_DIRECTORY = "tests/sparseml/transformers/finetune/finetune_custom/gpu"


class TestFinetuneNoRecipeCustomDataset(unittest.TestCase):
    def _test_finetune_wout_recipe_custom_dataset(self):
        from sparseml.transformers import train

        dataset_path = Path(tempfile.mkdtemp())

        created_success = self._create_mock_custom_dataset_folder_structure(
            dataset_path, self.file_extension
        )
        assert created_success

        def preprocessing_func(example):
            example["text"] = "Review: " + example["text"]
            return example

        concatenate_data = False

        train(
            model=self.model,
            dataset=self.file_extension,
            output_dir=self.output,
            recipe=None,
            num_train_epochs=self.num_train_epochs,
            concatenate_data=concatenate_data,
            oneshot_device=self.device,
            text_column="text",
            dataset_path=dataset_path,
            preprocessing_func=preprocessing_func,
            precision="bfloat16",
            bf16=True,
        )

    def _create_mock_custom_dataset_folder_structure(
        self, tmp_dir_data, file_extension
    ):
        train_path = os.path.join(tmp_dir_data, "train")
        test_path = os.path.join(tmp_dir_data, "test")
        validate_path = os.path.join(tmp_dir_data, "validate")

        # create tmp mock data files
        self.create_mock_file(
            extension=file_extension,
            content="text for train data 1",
            path=train_path,
            filename="data1",
        )
        self.create_mock_file(
            extension=file_extension,
            content="text for train data 2",
            path=train_path,
            filename="data2",
        )
        self.create_mock_file(
            extension=file_extension,
            content="text for test data 1",
            path=test_path,
            filename="data3",
        )
        self.create_mock_file(
            extension=file_extension,
            content="text for validate data 1",
            path=validate_path,
            filename="data4",
        )
        return True

    def create_mock_file(self, extension, content, path, filename):
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

    def tearDown(self):
        shutil.rmtree(self.output)


@requires_torch
@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneshotCustomDatasetSmall(TestFinetuneNoRecipeCustomDataset):
    model = None  # "Xenova/llama2.c-stories15M"
    file_extension = None  # ["json", "csv"]
    num_train_epochs = None

    def setUp(self):
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"

    def test_oneshot_then_finetune_small(self):
        self._test_finetune_wout_recipe_custom_dataset()


@requires_torch
@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestOneshotCustomDatasetGPU(TestFinetuneNoRecipeCustomDataset):
    model = None
    file_extension = None
    num_train_epochs = None

    def setUp(self):
        import torch

        from sparseml.transformers import SparseAutoModelForCausalLM

        self.device = "auto"
        self.output = "./oneshot_output"

        if "zoo:" in self.model:
            self.model = SparseAutoModelForCausalLM.from_pretrained(
                self.model, device_map=self.device, torch_dtype=torch.bfloat16
            )

    def test_oneshot_then_finetune_gpu(self):
        self._test_finetune_wout_recipe_custom_dataset()
