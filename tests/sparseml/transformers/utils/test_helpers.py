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

import pytest
import torch

from huggingface_hub import snapshot_download
from sparseml.transformers.utils.helpers import (
    is_transformer_generative_model,
    is_transformer_model,
    run_transformers_inference,
    save_zoo_directory,
)
from sparseml.transformers.utils.initializers import initialize_config, initialize_model
from sparsezoo import Model


@pytest.fixture()
def generative_model_path(tmp_path):
    return snapshot_download("roneneldan/TinyStories-1M", local_dir=tmp_path)


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
def dummy_inputs():
    input_ids = torch.zeros((1, 32), dtype=torch.int64)
    attention_mask = torch.ones((1, 32), dtype=torch.int64)

    return OrderedDict(
        input_ids=input_ids,
        attention_mask=attention_mask,
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


def test_is_transformer_generative_model(generative_model_path):
    assert is_transformer_generative_model(generative_model_path)


def test_run_transformers_inference_generative(generative_model_path, dummy_inputs):
    config = initialize_config(
        model_path=generative_model_path,
        trust_remote_code=True,
        **dict(use_cache=False),
    )
    model = initialize_model(
        model_path=generative_model_path,
        task="text-generation",
        config=config,
    )

    inputs, label, output = run_transformers_inference(inputs=dummy_inputs, model=None)
    assert isinstance(inputs, dict)
    assert label is None
    assert output is None

    inputs, label, output = run_transformers_inference(inputs=dummy_inputs, model=model)
    assert isinstance(inputs, dict)
    assert label is None
    assert isinstance(output, dict)


def test_run_tranformers_inference(model_path, dummy_inputs):

    config = initialize_config(model_path=model_path, trust_remote_code=True)
    model = initialize_model(
        model_path=model_path,
        task="qa",
        config=config,
    )

    inputs, label, output = run_transformers_inference(inputs=dummy_inputs, model=None)
    assert isinstance(inputs, dict)
    assert label is None
    assert output is None

    inputs, label, output = run_transformers_inference(inputs=dummy_inputs, model=model)
    assert isinstance(inputs, dict)
    assert label is None
    assert isinstance(output, dict)


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
