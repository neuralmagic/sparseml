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
import re
import shutil

import pytest
import torch

from huggingface_hub import snapshot_download
from sparseml.pytorch.utils.helpers import default_device
from sparseml.transformers.utils.initializers import (
    initialize_config,
    initialize_sparse_model,
    initialize_tokenizer,
    initialize_trainer,
)
from sparseml.transformers.utils.load_task_dataset import load_task_dataset
from sparsezoo import Model


def save_recipe_for_text_classification(source_path):
    recipe = """test_stage:
             quant_modifiers:
               LegacyQuantizationModifier:
                 post_oneshot_calibration: False
                 scheme_overrides:
                   Embedding:
                     input_activations: null"""

    with open(os.path.join(source_path, "recipe.yaml"), "w") as f:
        f.write(recipe)


@pytest.mark.parametrize("device", ["auto", "cpu", None])
@pytest.mark.parametrize(
    "stub, task, data_args",
    [
        (
            "zoo:obert-medium-squad_wikipedia_bookcorpus-pruned95_quantized",
            "qa",
            dict(dataset_name="squad"),
        ),
        (
            "roneneldan/TinyStories-1M",
            "text-generation",
            None,
        ),
    ],
)
class TestInitializeModelFlow:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, stub, task, data_args, device):
        self.model_path = (
            Model(stub, tmp_path).training.path
            if stub.startswith("zoo:")
            else snapshot_download(stub, local_dir=tmp_path)
        )
        self.task = task
        self.data_args = data_args
        self.device = default_device() if device == "auto" else device

        if task == "text-classification":
            # for text classification, save a dummy recipe to the model path
            # so that it can be loaded
            save_recipe_for_text_classification(self.model_path)

        self.sequence_length = 384
        yield
        shutil.rmtree(tmp_path)

    def test_initialize_config(self, setup):
        assert initialize_config(model_path=self.model_path, trust_remote_code=True)

    def test_initialize_tokenizer(self, setup):
        tokenizer = initialize_tokenizer(
            self.model_path, self.sequence_length, self.task
        )
        if self.task == "text-generation":
            assert tokenizer.pad_token_id

        assert tokenizer.model_max_length == self.sequence_length

    def test_initialize_model(self, setup):
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        model = initialize_sparse_model(
            model_path=self.model_path,
            task=self.task,
            device=self.device,
            recipe=None,
            config=config,
        )
        self._test_model_device(model)

    def test_initialize_trainer(self, setup):
        if not self.data_args:
            pytest.skip("To run this test, please provide valid data_args")
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        model = initialize_sparse_model(
            model_path=self.model_path,
            task=self.task,
            device=self.device,
            recipe=None,
            config=config,
        )
        if isinstance(model, torch.nn.DataParallel):
            pytest.skip("Cannot initialize a trainer for a DataParallel model")
        tokenizer = initialize_tokenizer(
            self.model_path, self.sequence_length, self.task
        )
        dataset = load_task_dataset(
            task=self.task,
            tokenizer=tokenizer,
            data_args=self.data_args,
            model=model,
            config=config,
        )

        trainer = initialize_trainer(
            model=model, model_path=self.model_path, validation_dataset=dataset
        )
        # assert that trainer is not messing up with model's location
        self._test_model_device(model)
        assert trainer.get_eval_dataloader()

    def test_initialize_trainer_no_validation_dataset(self, setup):
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        tokenizer = initialize_tokenizer(
            self.model_path, self.sequence_length, self.task
        )
        model = initialize_sparse_model(
            model_path=self.model_path,
            task=self.task,
            device=self.device,
            recipe=None,
            config=config,
        )
        if isinstance(model, torch.nn.DataParallel):
            pytest.skip("Cannot initialize a trainer for a DataParallel model")
        trainer = initialize_trainer(
            model=model, model_path=self.model_path, validation_dataset=None
        )
        self._test_model_device(model)
        assert trainer.eval_dataset is None
        assert trainer._get_fake_dataloader(num_samples=10, tokenizer=tokenizer)

    def _test_model_device(self, model):
        if self.device is None or self.device == "cpu":
            assert model.device.type == "cpu"
            return
        use_multiple_gpus = re.match(r"cuda:\d+,(\d+)*", self.device)
        if use_multiple_gpus:
            assert isinstance(model, torch.nn.DataParallel)
        else:
            assert model.device.type == "cuda"
