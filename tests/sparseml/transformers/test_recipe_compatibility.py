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

import shutil

import pytest

import sparseml
from huggingface_hub import snapshot_download
from sparseml.transformers import SparseAutoModelForCausalLM


@pytest.fixture
def model_path(tmp_path):
    yield snapshot_download("stas/tiny-random-llama-2", local_dir=tmp_path)
    shutil.rmtree(tmp_path)


@pytest.fixture
def recipe():
    return """test_stage:
  obcq_modifiers:
    LegacyQuantizationModifier:
      ignore:
        - LlamaRotaryEmbedding
        - LlamaRMSNorm
        - {silu_activation}
      scheme_overrides:
        Embedding:
          input_activations: null
          weights:
            num_bits: 8
            symmetric: false"""


def test_silu_alias_same_output(recipe, model_path):
    model_ = SparseAutoModelForCausalLM.from_pretrained(
        model_path, recipe=recipe.format(silu_activation="SiLU")
    )
    sparseml.create_session()
    sparseml.reset_session()
    model = SparseAutoModelForCausalLM.from_pretrained(
        model_path, recipe=recipe.format(silu_activation="SiLUActivation")
    )

    dummy_input = model.dummy_inputs

    out = model(**dummy_input)
    out_ = model_(**dummy_input)

    out.logits.allclose(out_.logits)
