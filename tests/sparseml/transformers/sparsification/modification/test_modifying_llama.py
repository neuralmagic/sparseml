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

from sparseml.transformers.sparsification.modification.modifying_llama import (
    LlamaAttentionWithQuantizableMatmuls,
)


@pytest.fixture
def llama_recipe():
    return """test_stage:
  quant_modifiers:
    LegacyQuantizationModifier:
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


def test_modify_with_quantization_recipe(
    llama_model, llama_recipe, shared_helper_functions
):
    shared_helper_functions.check_model_modified_causal(
        llama_model,
        recipe=llama_recipe,
        modified_module=LlamaAttentionWithQuantizableMatmuls,
    )


def test_modify_with_quantization_recipe_sparsezoo(
    llama_zoo_model, llama_recipe, shared_helper_functions
):
    # TODO: Improve that
    shared_helper_functions.check_model_modified_causal(
        llama_zoo_model,
        recipe=llama_recipe,
        modified_module=LlamaAttentionWithQuantizableMatmuls,
    )
