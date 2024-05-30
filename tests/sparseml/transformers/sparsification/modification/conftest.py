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

from copy import deepcopy

import pytest
from transformers import AutoConfig, AutoModel

from accelerate import init_empty_weights
from sparseml.modifiers.quantization_legacy.modification import modify_model
from sparseml.pytorch.model_load.helpers import apply_recipe_structure_to_model
from sparseml.transformers import SparseAutoConfig, SparseAutoModelForCausalLM


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
def mobilebert_model():
    config = AutoConfig.from_pretrained("google/mobilebert-uncased")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def opt_zoo_model():
    stub = "zoo:opt-1.3b-opt_pretrain-quantW8A8"
    config = SparseAutoConfig.from_pretrained(stub)
    with init_empty_weights():
        model = SparseAutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture
def opt_model():
    config = AutoConfig.from_pretrained("facebook/opt-1.3b")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def mistral_zoo_model():
    stub = "zoo:mistral-7b-evolcodealpaca_mistral_pretrain-pruned50_quantized"
    config = SparseAutoConfig.from_pretrained(stub)
    with init_empty_weights():
        model = SparseAutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture
def llama_zoo_model():
    stub = "zoo:llama2-7b-llama2_chat_llama2_pretrain-base_quantized"
    config = SparseAutoConfig.from_pretrained(stub)
    with init_empty_weights():
        model = SparseAutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture
def mistral_model():
    config = AutoConfig.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def llama_model():
    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


@pytest.fixture
def shared_helper_functions():
    return SharedHelperFunctions


class SharedHelperFunctions:
    @staticmethod
    def check_model_modified_non_causal(
        original_model_, modified_module, num_modified_modules=None
    ):

        num_attn_blocks = original_model_.config.num_hidden_layers
        num_modified_modules = num_modified_modules or num_attn_blocks
        original_model = deepcopy(original_model_)
        modified_model = modify_model(original_model_)

        # make sure that the original model has 0 modified modules
        # and that the modified model has N modified modules
        # where N is the number of transformer's attention blocks
        assert (
            sum(
                [
                    module.__class__.__name__ == modified_module.__name__
                    for module in modified_model.modules()
                ]
            )
            == num_modified_modules
        )
        assert (
            sum(
                [
                    module.__class__.__name__ == modified_module.__name__
                    for module in original_model.modules()
                ]
            )
            == 0
        )

    @staticmethod
    def check_model_modified_causal(
        original_model_,
        modified_module,
        recipe,
    ):
        num_attn_blocks = original_model_.config.num_hidden_layers

        original_model = deepcopy(original_model_)
        modified_model = original_model_

        apply_recipe_structure_to_model(
            model=modified_model,
            model_path=None,
            recipe_path=recipe,
        )

        # make sure that the original model has 0 modified modules
        # and that the modified model has N modified modules
        # where N is the number of transformer's attention blocks
        assert (
            sum(
                [
                    module.__class__.__name__ == modified_module.__name__
                    for module in modified_model.modules()
                ]
            )
            == num_attn_blocks
        )
        assert (
            sum(
                [
                    module.__class__.__name__ == modified_module.__name__
                    for module in original_model.modules()
                ]
            )
            == 0
        )
