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

# TODO: Those tests should be hardened

import pytest

from sparsezoo import Model
from src.sparseml.transformers.refactor_utils.initialize_model import (
    initialize_transformer_model,
)


@pytest.fixture()
def model_path():
    return Model(
        "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized"
    ).training.path


@pytest.fixture()
def sequence_length():
    return 384


@pytest.fixture()
def task():
    return "qa"


def test_initialize_transformer_model(model_path, sequence_length, task):
    model, trainer, config, tokenizer = initialize_transformer_model(
        model_path=model_path, sequence_length=sequence_length, task=task
    )
    assert model.base_model_prefix == config.model_type == "mobilebert"
    assert trainer
    assert tokenizer.model_max_length == sequence_length
