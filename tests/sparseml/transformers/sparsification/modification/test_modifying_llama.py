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


def test_modifying_llama(llama_model):
    from sparseml.transformers.sparsification.modification.modifying_llama import (  # noqa F401
        modify,
    )

    num_attn_blocks = llama_model.config.num_hidden_layers

    # keep the original model for comparison
    llama_ = deepcopy(llama_model)
    llama = modify_model(llama_model)

    # check how many modified "LlamalAttention" modules are in the original
    # model (should be 0, as the model is not modified yet)
    modified_modules_original_model = [
        module
        for module in llama_.modules()
        if _is_llama_attention_modified(module)
        and module.__class__.__name__ == "LlamaAttention"
    ]
    # check how many modified "LLamalAttention" modules are
    # in the modified model (should be num_attn_blocks, as the
    # model is modified, and has num_attn_blocks attention blocks)
    modified_modules_modified_model = [
        module
        for module in llama.modules()
        if _is_llama_attention_modified(module)
        and module.__class__.__name__ == "LlamaAttention"
    ]
    # check how many original "LlamalAttention"
    # modules are in the original
    # model (should be num_attn_blocks, as the model is
    # not modified yet, and has num_attn_blocks attention blocks)
    original_modules_original_model = [
        module
        for module in llama_.modules()
        if not _is_llama_attention_modified(module)
        and module.__class__.__name__ == "LlamaAttention"
    ]
    # check how many original "LlamalAttention"
    # modules are in the modified
    # model (should be 0, as the model is
    # modified, and should not contain any original
    # "LlamalAttention" modules)
    original_modules_modified_model = [
        module
        for module in llama.modules()
        if not _is_llama_attention_modified(module)
        and module.__class__.__name__ == "LlamaAttention"
    ]

    assert (
        len(modified_modules_original_model)
        == len(original_modules_modified_model)
        == 0
    )
    assert (
        len(modified_modules_modified_model)
        == len(original_modules_original_model)
        == num_attn_blocks
    )


def test_apply_recipe_fail(llama_recipe, llama_zoo_model):
    from sparseml.transformers.sparsification.modification.modifying_llama import (  # noqa F401
        modify,
    )

    with pytest.raises(Exception):
        apply_recipe_structure_to_model(
            model=llama_zoo_model, model_path=None, recipe_path=llama_recipe
        )


def test_apply_recipe(llama_recipe, llama_zoo_model):
    from sparseml.transformers.sparsification.modification.modifying_llama import (  # noqa F401
        modify,
    )

    apply_recipe_structure_to_model(
        model=modify_model(llama_zoo_model), model_path=None, recipe_path=llama_recipe
    )
    assert True


def _is_llama_attention_modified(module):
    # only the modified "LlamaAttention"
    # modules have the "attn_output_matmul" attribute
    return hasattr(module, "attn_output_matmul")
