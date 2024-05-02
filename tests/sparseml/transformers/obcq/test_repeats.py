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

import math

import pytest
import torch
import yaml

import sparseml
from sparseml.pytorch.model_load.helpers import get_session_model
from sparseml.pytorch.utils.helpers import tensor_sparsity
from sparseml.transformers import oneshot
from sparseml.utils.pytorch import qat_active


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


def test_consecutive_runs(tmp_path):
    tiny_model_path = "Xenova/llama2.c-stories15M"
    first_recipe = "tests/sparseml/transformers/obcq/quant_and_sparse.yaml"
    second_recipe = "tests/sparseml/transformers/obcq/additional_sparsity.yaml"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    # test recipe with 50% sparsity, quantization and smoothquant
    oneshot(
        model=tiny_model_path,
        dataset="open_platypus",
        num_calibration_samples=16,
        recipe=first_recipe,
        output_dir=tmp_path / "test1",
        oneshot_device=device,
        clear_sparse_session=False,
    )
    first_tiny_model = get_session_model()
    layer_0_sparse = tensor_sparsity(
        first_tiny_model.model.layers[0].self_attn.k_proj.module.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.5, rel_tol=1e-3)
    assert qat_active(first_tiny_model)

    session = sparseml.active_session()
    session_recipe = session.lifecycle.recipe_container.compiled_recipe
    stages = [stage.group for stage in session_recipe.stages]
    assert len(stages) == 1
    session.reset()

    # reload saved model and up sparsity to 0.7
    oneshot(
        model=tmp_path / "test1",
        dataset="open_platypus",
        num_calibration_samples=16,
        recipe=second_recipe,
        output_dir=tmp_path / "test2",
        oneshot_device=device,
        clear_sparse_session=False,
    )

    second_tiny_model = get_session_model()
    layer_0_sparse = tensor_sparsity(
        second_tiny_model.model.layers[0].self_attn.k_proj.module.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.7, rel_tol=1e-3)
    assert qat_active(second_tiny_model)

    session = sparseml.active_session()
    session_recipe = session.lifecycle.recipe_container.compiled_recipe
    stages = [stage.group for stage in session_recipe.stages]
    assert len(stages) == 2

    recipe_path = tmp_path / "test2" / "recipe.yaml"
    recipe_data = yaml.safe_load(recipe_path.read_text())
    stage_keys = recipe_data.keys()
    assert len(stage_keys) == 2
    assert "test_stage_0" in stage_keys
    assert "test_stage_1" in stage_keys


def test_fail_on_repeated_quant(tmp_path):
    first_recipe_str = """
    first_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLU
                scheme_overrides:
                    Embedding:
                        input_activations: null
    """

    second_recipe_str = """
    second_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLU
                    - Embedding
    """

    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    oneshot(
        model=tiny_model_path,
        dataset="open_platypus",
        num_calibration_samples=4,
        oneshot_device=device,
        recipe=first_recipe_str,
        output_dir=tmp_path / "test",
        clear_sparse_session=False,
    )

    session = sparseml.active_session()
    session.reset()

    # When trying to re-quantize with the second recipe, we should error out
    # to avoid nested quantizations
    with pytest.raises(RuntimeError):
        oneshot(
            model=tmp_path / "test",
            dataset="open_platypus",
            num_calibration_samples=4,
            oneshot_device=device,
            recipe=second_recipe_str,
        )


def test_separate_quants_allowed(tmp_path):
    first_recipe_str = """
    first_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLU
                    - Linear
                scheme_overrides:
                    Embedding:
                        input_activations: null
    """

    second_recipe_str = """
    second_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLU
                    - Embedding
                    - MatMulLeftInput_QK
                    - MatMulRightInput_QK
                    - MatMulOutput_QK
                    - MatMulLeftInput_PV
                    - MatMulRightInput_PV
                    - MatMulOutput_PV
                    - QuantizableMatMul
    """

    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    oneshot(
        model=tiny_model_path,
        dataset="open_platypus",
        num_calibration_samples=16,
        recipe=first_recipe_str,
        output_dir=tmp_path / "test1",
        oneshot_device=device,
        clear_sparse_session=False,
    )
    # only embedding quantized after first recipe
    first_model = get_session_model()
    assert not isinstance(
        first_model.model.layers[0].mlp.down_proj, torch_quantization.QuantWrapper
    )
    assert hasattr(first_model.model.embed_tokens, "quantization_scheme")
    session = sparseml.active_session()
    session.reset()

    # When trying to re-quantize with the second recipe, we should error out
    # to avoid nested quantizations
    oneshot(
        model=tmp_path / "test1",
        dataset="open_platypus",
        num_calibration_samples=16,
        recipe=second_recipe_str,
        output_dir=tmp_path / "test2",
        oneshot_device=device,
        clear_sparse_session=False,
    )

    second_model = get_session_model()
    # linear and embeddings should be quantized now
    assert isinstance(
        second_model.model.layers[0].mlp.down_proj, torch_quantization.QuantWrapper
    )
    assert hasattr(second_model.model.embed_tokens, "quantization_scheme")
