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

from sparsezoo import Model
from src.sparseml.transformers.utils.create_model import (
    create_model,
    initialize_config,
    initialize_tokenizer,
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
class TestCreateModelFlow:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task, data_args):
        self.model_path = Model(stub, tmp_path).training.path
        self.sequence_length = 384
        self.task = task
        self.data_args = data_args

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

    def test_initialize_config(self, setup):
        assert initialize_config(model_path=self.model_path, trust_remote_code=True)

    def test_create_model(self, setup):
        if self.data_args is None:
            pytest.skip("No data args provided")

        model, *_, validation = create_model(
            model_path=self.model_path,
            task=self.task,
            trust_remote_code=True,
            data_args=self.data_args,
        )
        assert validation is not None

    def test_create_model_no_data_args(self, setup):
        model, *_, validation = create_model(
            model_path=self.model_path,
            task=self.task,
            trust_remote_code=True,
            data_args=None,
        )
        assert validation is None
