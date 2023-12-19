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

import shutil

import pytest

from sparseml.pytorch.utils.helpers import default_device, use_single_gpu
from sparseml.transformers.utils.load_task_dataset import load_task_dataset
from sparsezoo import Model
from src.sparseml.transformers.utils.initializers import (
    initialize_config,
    initialize_model,
    initialize_tokenizer,
    initialize_trainer,
)


@pytest.mark.parametrize(
    "stub, task, data_args",
    [
        (
            "zoo:obert-medium-squad_wikipedia_bookcorpus-pruned95_quantized",
            "qa",
            dict(dataset_name="squad"),
        ),
        (
            "zoo:distilbert-qqp_wikipedia_bookcorpus-pruned80.4block_quantized",
            "text-classification",
            None,
        ),
    ],
    scope="class",
)
@pytest.mark.parametrize("device", ["auto", "cpu", None], scope="class")
class TestInitializeModelFlow:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task, data_args, device):
        self.model_path = Model(stub, tmp_path).training.path
        self.sequence_length = 384
        self.task = task
        self.data_args = data_args

        # process device argument
        device = default_device() if device == "auto" else device
        # if multiple gpus available use the first one
        if not (device is None or device == "cpu"):
            device = use_single_gpu(device)
        self.device = device
        yield
        shutil.rmtree(tmp_path)

    def test_initialize_config(self, setup):
        assert initialize_config(model_path=self.model_path, trust_remote_code=True)

    def test_initialize_tokenizer(self, setup):
        tokenizer = initialize_tokenizer(
            self.model_path, self.sequence_length, self.task
        )
        assert (
            tokenizer.padding_side == "right"
            if self.task != "text-generation"
            else "left"
        )
        assert tokenizer.model_max_length == self.sequence_length

    def test_initialize_model(self, setup):
        model = initialize_model(
            model_path=self.model_path,
            task=self.task,
            device=self.device,
            config=initialize_config(
                model_path=self.model_path, trust_remote_code=True
            ),
        )
        assert model
        self._test_model_device(model)

    def test_initialize_trainer(self, setup):
        if not self.data_args:
            pytest.skip("To run this test, please provide valid data_args")
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        model = initialize_model(
            model_path=self.model_path,
            task=self.task,
            device=self.device,
            config=config,
        )
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
        validation_dataset = dataset.get("validation")

        trainer = initialize_trainer(
            model=model,
            model_path=self.model_path,
            validation_dataset=validation_dataset,
        )
        # assert that trainer is not messing up with model's location
        self._test_model_device(model)
        assert trainer.get_eval_dataloader()

    def test_initialize_trainer_no_validation_dataset(self, setup):
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        tokenizer = initialize_tokenizer(
            self.model_path, self.sequence_length, self.task
        )
        model = initialize_model(
            model_path=self.model_path,
            task=self.task,
            config=config,
        )
        trainer = initialize_trainer(
            model=model, model_path=self.model_path, validation_dataset=None
        )
        self._test_model_device(model)
        assert trainer.eval_dataset is None
        assert trainer._get_fake_dataloader(num_samples=10, tokenizer=tokenizer)

    def _test_model_device(self, model):
        if model.device.type == "cuda":
            assert self.device.startswith("cuda")
        else:
            assert self.device is None or self.device == "cpu"
