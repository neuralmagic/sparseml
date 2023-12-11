# apply recipe structure

# reload model state
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

from transformers import AutoModelForCausalLM

from sparseml.pytorch.model_load.helpers import apply_recipe_structure_to_model
from sparseml.utils.pytorch.module import qat_active


def test_apply_recipe_structure():
    model_path = "Xenova/llama2.c-stories15M"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    assert not qat_active(model)

    recipe_with_quant = "tests/sparseml/transformers/obcq/test_tiny.yaml"
    apply_recipe_structure_to_model(model, recipe_with_quant, model_path)

    assert qat_active(model)
