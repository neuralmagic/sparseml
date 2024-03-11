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
from transformers.models.llama.modeling_llama import LlamaAttention

from sparseml.pytorch.model_load.helpers import apply_recipe_structure_to_model
from sparseml.transformers.sparsification.modification import modify_model


@pytest.fixture
def llama_recipe():
    return """test_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore:
        - MatMulRightInput_QK
        - MatMulLeftInput_QK
        - MatMulRightInput_PV
        - MatMulLeftInput_PV
        - MatMulOutput_QK
        - MatMulOutput_PV
      scheme_overrides:
        Embedding:
          input_activations: null
          weights:
            num_bits: 8
            symmetric: False"""


def test_modifying_llama(llama_model, shared_helper_functions):

    shared_helper_functions.check_model_modified(
        llama_model,
        module_to_replace=LlamaAttention,
        func_to_validate_replacement=_is_llama_attention_modified,
    )


def test_apply_recipe_fail(llama_recipe, llama_zoo_model):

    with pytest.raises(Exception):
        apply_recipe_structure_to_model(
            model=llama_zoo_model, model_path=None, recipe_path=llama_recipe
        )


def test_apply_recipe(llama_recipe, llama_zoo_model):
    apply_recipe_structure_to_model(
        model=modify_model(llama_zoo_model), model_path=None, recipe_path=llama_recipe
    )
    assert True


def _is_llama_attention_modified(module):
    # only the modified "LlamaAttention"
    # modules have the "attn_output_matmul" attribute
    return hasattr(module, "attn_output_matmul")
