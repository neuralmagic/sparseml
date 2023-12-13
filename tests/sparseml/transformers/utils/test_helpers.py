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

from sparseml.transformers.utils.helpers import (
    create_dummy_inputs,
    is_transformer_model,
    save_zoo_directory,
)
from sparsezoo import Model
from src.sparseml.transformers.utils.model import SparseAutoModel


@pytest.fixture()
def model_path(tmp_path):
    return Model(
        "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized",
        tmp_path,
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
@pytest.mark.parametrize(
    "batch_size",
    [1, 10],
)
def test_create_dummy_inputs(
    model_path, sequence_length, inputs_type, expected_dummy_inputs, batch_size
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, model_max_length=sequence_length
    )
    model = SparseAutoModel.question_answering_from_pretrained(
        model_name_or_path=model_path, model_type="model"
    )
    dummy_inputs = create_dummy_inputs(
        model, tokenizer, type=inputs_type, batch_size=batch_size
    )
    for key in dummy_inputs:
        input = dummy_inputs[key]
        assert numpy.array_equal(
            input.numpy() if inputs_type == "pt" else input, expected_dummy_inputs[key]
        )


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
    ],
)
def test_is_transformer_model(tmp_path, stub):
    zoo_model = Model(stub, tmp_path)
    source_path = zoo_model.training.path
    assert is_transformer_model(source_path)


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
        "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none",  # noqa E501
    ],
)
def test_save_zoo_directory(stub, tmp_path_factory):
    path_to_training_outputs = tmp_path_factory.mktemp("outputs")
    save_dir = tmp_path_factory.mktemp("save_dir")

    zoo_model = Model(stub, path_to_training_outputs)
    zoo_model.download()

    save_zoo_directory(
        output_dir=save_dir,
        training_outputs_dir=path_to_training_outputs,
    )
    new_zoo_model = Model(str(save_dir))
    assert new_zoo_model.validate(minimal_validation=True, validate_onnxruntime=False)
