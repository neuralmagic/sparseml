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

import sparseml.core.session as session_manager
from sparseml.pytorch.utils.helpers import tensor_sparsity
from sparseml.transformers.sparsification.obcq.obcq import one_shot
from sparseml.utils.pytorch import qat_active


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


def test_consecutive_runs(tmp_path):
    tiny_model_path = "Xenova/llama2.c-stories15M"
    first_recipe = "tests/sparseml/transformers/obcq/test_tiny.yaml"
    second_recipe = "tests/sparseml/transformers/obcq/test_additional_sparsity.yaml"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    # test recipe with 50% sparsity, quantization and smoothquant
    first_tiny_model = one_shot(
        model_path=tiny_model_path,
        dataset_name="open_platypus",
        num_samples=16,
        device=device,
        recipe_file=first_recipe,
        deploy_dir=tmp_path / "test1",
        do_save=True,
    )
    layer_0_sparse = tensor_sparsity(
        first_tiny_model.model.layers[0].self_attn.k_proj.module.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.5, rel_tol=1e-3)
    assert qat_active(first_tiny_model)

    session = session_manager.active_session()
    session_recipe = session.lifecycle.recipe_container.compiled_recipe
    stages = [stage.group for stage in session_recipe.stages]
    assert len(stages) == 1
    session.reset()

    # reload saved model and up sparsity to 0.7
    second_tiny_model = one_shot(
        model_path=tmp_path / "test1" / "obcq_deployment",
        dataset_name="open_platypus",
        num_samples=16,
        device=device,
        recipe_file=second_recipe,
        deploy_dir=tmp_path / "test2",
        do_save=True,
    )
    layer_0_sparse = tensor_sparsity(
        second_tiny_model.model.layers[0].self_attn.k_proj.module.weight
    )
    assert math.isclose(layer_0_sparse.item(), 0.7, rel_tol=1e-3)
    assert qat_active(second_tiny_model)

    session = session_manager.active_session()
    session_recipe = session.lifecycle.recipe_container.compiled_recipe
    stages = [stage.group for stage in session_recipe.stages]
    assert len(stages) == 2

    recipe_path = tmp_path / "test2" / "obcq_deployment" / "recipe.yaml"
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
                    - SiLUActivation
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
                    - SiLUActivation
                    - Embedding
    """

    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    one_shot(
        model_path=tiny_model_path,
        dataset_name="open_platypus",
        num_samples=4,
        device=device,
        recipe_file=first_recipe_str,
        deploy_dir=tmp_path,
        do_save=True,
    )

    session = session_manager.active_session()
    session.reset()

    # When trying to re-quantize with the second recipe, we should error out
    # to avoid nested quantizations
    with pytest.raises(RuntimeError):
        one_shot(
            model_path=tmp_path / "obcq_deployment",
            dataset_name="open_platypus",
            num_samples=4,
            device=device,
            recipe_file=second_recipe_str,
        )


def test_separate_quants_allowed(tmp_path):
    first_recipe_str = """
    first_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLUActivation
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
                    - SiLUActivation
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

    first_model = one_shot(
        model_path=tiny_model_path,
        dataset_name="open_platypus",
        num_samples=4,
        device=device,
        recipe_file=first_recipe_str,
        deploy_dir=tmp_path,
        do_save=True,
    )

    # only embedding quantized after first recipe
    assert not isinstance(
        first_model.model.layers[0].mlp.down_proj, torch_quantization.QuantWrapper
    )
    assert hasattr(first_model.model.embed_tokens, "quantization_scheme")
    session = session_manager.active_session()
    session.reset()

    # When trying to re-quantize with the second recipe, we should error out
    # to avoid nested quantizations
    second_model = one_shot(
        model_path=tmp_path / "obcq_deployment",
        dataset_name="open_platypus",
        num_samples=4,
        device=device,
        recipe_file=second_recipe_str,
    )

    # linear and embeddings should be quantized now
    assert isinstance(
        second_model.model.layers[0].mlp.down_proj, torch_quantization.QuantWrapper
    )
    assert hasattr(second_model.model.embed_tokens, "quantization_scheme")
