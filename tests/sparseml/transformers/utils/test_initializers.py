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

import pytest

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
class TestInitializeModelFlow:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task, data_args):
        self.model_path = Model(stub, tmp_path).training.path
        self.sequence_length = 384
        self.task = task
        self.data_args = data_args

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
        assert initialize_model(
            model_path=self.model_path,
            task=self.task,
            config=initialize_config(
                model_path=self.model_path, trust_remote_code=True
            ),
        )

    def test_initialize_trainer(self, setup):
        if not self.data_args:
            pytest.skip("To run this test, please provide valid data_args")
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        model = initialize_model(
            model_path=self.model_path,
            task=self.task,
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

        assert initialize_trainer(
            model=model,
            model_path=self.model_path,
            validation_dataset=validation_dataset,
        )

    def test_initialize_trainer_no_validation_dataset(self, setup):
        config = initialize_config(model_path=self.model_path, trust_remote_code=True)
        model = initialize_model(
            model_path=self.model_path,
            task=self.task,
            config=config,
        )
        assert initialize_trainer(
            model=model, model_path=self.model_path, validation_dataset=None
        )
