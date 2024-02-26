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
from transformers import AutoConfig, AutoModel

from accelerate import init_empty_weights
from sparseml.transformers import (
    SparseAutoConfig,
    SparseAutoModel,
    SparseAutoModelForCausalLM,
)
from sparsezoo import Model


@pytest.fixture
def mistral_zoo_model():
    stub = "zoo:mistral-7b-evolcodealpaca_mistral_pretrain-pruned50_quantized"
    config = SparseAutoConfig.from_pretrained(stub)
    with init_empty_weights():
        model = SparseAutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture
def opt_zoo_model():
    stub = "zoo:opt-1.3b-opt_pretrain-quantW8A8"
    config = SparseAutoConfig.from_pretrained(stub)
    with init_empty_weights():
        model = SparseAutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture
def llama_zoo_model():
    stub = "zoo:llama2-7b-llama2_chat_llama2_pretrain-base_quantized"
    config = SparseAutoConfig.from_pretrained(stub)
    with init_empty_weights():
        # attn_implementation="eager" needed so that the model uses the
        # appropriate attention type
        model = SparseAutoModelForCausalLM.from_config(
            config, attn_implementation="eager"
        )
    return model


@pytest.fixture
def distilbert_zoo_model(tmp_path):
    stub = "zoo:distilbert-squad_wikipedia_bookcorpus-pruned80.4block_quantized"
    model_path = Model(stub, tmp_path).training.path
    model = SparseAutoModel.question_answering_from_pretrained(
        model_path, model_type="model"
    )
    return model


@pytest.fixture
def mobilebert_zoo_model(tmp_path):
    stub = "zoo:mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized"
    model_path = Model(stub, tmp_path).training.path
    model = SparseAutoModel.question_answering_from_pretrained(
        model_path, model_type="model"
    )
    return model


@pytest.fixture
def bert_zoo_model(tmp_path):
    stub = "zoo:bert-base-squad_wikipedia_bookcorpus-pruned95.obs_quantized"
    model_path = Model(stub, tmp_path).training.path
    model = SparseAutoModel.question_answering_from_pretrained(
        model_path, model_type="model"
    )
    return model


@pytest.fixture
def bert_model():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def distilbert_model():
    config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def mistral_model():
    config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def mobilebert_model():
    config = AutoConfig.from_pretrained("google/mobilebert-uncased")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def llama_model():
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    with init_empty_weights():
        # attn_implementation="eager" needed so that the model uses the
        # appropriate attention type
        model = AutoModel.from_config(config, attn_implementation="eager")
    return model


@pytest.fixture
def opt_model():
    config = AutoConfig.from_pretrained("facebook/opt-1.3b")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model
