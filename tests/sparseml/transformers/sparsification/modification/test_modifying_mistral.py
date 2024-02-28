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
from transformers.models.mistral.modeling_mistral import MistralAttention

from sparseml.pytorch.model_load.helpers import apply_recipe_structure_to_model
from sparseml.transformers.sparsification.modification import modify_model


@pytest.fixture
def mistral_recipe():
    return """test_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore:
        - MatMulRightInput_QK
        - MatMulLeftInput_QK
        - MatMulRightInput_PV
        - MatMulLeftInput_PV
      scheme_overrides:
        Embedding:
          input_activations: null
          weights:
            num_bits: 8
            symmetric: False"""


def test_modifying_mistral(mistral_model, helpers):
    from sparseml.transformers.sparsification.modification.modifying_mistral import (  # noqa F401
        modify,
    )

    helpers.check_model_modified(
        mistral_model,
        module_to_replace=MistralAttention,
        func_to_validate_replacement=_is_mistral_attention_modified,
    )


def test_apply_recipe_fail(mistral_recipe, mistral_zoo_model):
    from sparseml.transformers.sparsification.modification.modifying_mistral import (  # noqa F401
        modify,
    )

    with pytest.raises(Exception):
        apply_recipe_structure_to_model(
            model=mistral_zoo_model, model_path=None, recipe_path=mistral_recipe
        )


def test_apply_recipe(mistral_recipe, mistral_zoo_model):
    from sparseml.transformers.sparsification.modification.modifying_mistral import (  # noqa F401
        modify,
    )

    apply_recipe_structure_to_model(
        model=modify_model(mistral_zoo_model),
        model_path=None,
        recipe_path=mistral_recipe,
    )
    assert True


def _is_mistral_attention_modified(module):
    # only the modified "MistralAttention"
    # modules have the "attn_output_matmul" attribute
    return hasattr(module, "attn_output_matmul")
