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

from collections import OrderedDict

import numpy
import pytest
from transformers import AutoTokenizer

from sparsezoo import Model
from src.sparseml.transformers.refactor_utils.create_dummy_inputs import (
    create_dummy_inputs,
)
from src.sparseml.transformers.utils.model import SparseAutoModel


@pytest.fixture()
def model_path():
    return Model(
        "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized"
    ).training.path


@pytest.fixture()
def sequence_length():
    return 384


@pytest.fixture()
def expected_dummy_inputs():
    input_ids = numpy.zeros((1, 384), dtype=numpy.int8)
    attention_mask = numpy.zeros((1, 384), dtype=numpy.int8)
    token_type_ids = numpy.zeros((1, 384), dtype=numpy.int8)

    input_ids[:, 0] = 101
    input_ids[:, 1] = 102
    attention_mask[:, :2] = 1

    return OrderedDict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )


@pytest.mark.parametrize(
    "inputs_type",
    ["pt", "np"],
)
def test_create_dummy_inputs(
    model_path, sequence_length, inputs_type, expected_dummy_inputs
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, model_max_length=sequence_length
    )
    model = SparseAutoModel.question_answering_from_pretrained(
        model_name_or_path=model_path, model_type="model"
    )
    dummy_inputs = create_dummy_inputs(model, tokenizer, type=inputs_type)
    for key in dummy_inputs:
        input = dummy_inputs[key]
        assert numpy.array_equal(
            input.numpy() if inputs_type == "pt" else input, expected_dummy_inputs[key]
        )
