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

from sparseml.core import ModifiableModel
from sparseml.core.framework import Framework
from sparseml.core.state import State
from sparseml.modifiers.obcq import SparseGPTModifier
from sparseml.modifiers.obcq.pytorch import SparseGPTModifierPyTorch
from sparseml.pytorch.model_load.helpers import get_session_model
from sparseml.pytorch.utils.helpers import tensor_sparsity
from sparseml.transformers import SparseAutoModelForCausalLM, oneshot


@pytest.mark.parametrize(
    "recipe_file_path",
    [
        "tests/sparseml/transformers/obcq/sparse.yaml",
        "tests/sparseml/transformers/obcq/quant.yaml",
        "tests/sparseml/transformers/obcq/quant_and_sparse.yaml",
    ],
)
def test_obcq_tinystories(recipe_file_path):
    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    oneshot(
        model=tiny_model_path,
        dataset="open_platypus",
        oneshot_device=device,
        recipe=recipe_file_path,
        max_seq_length=128,
        num_calibration_samples=64,
        pad_to_max_length=False,
    )


def test_lm_head_target():
    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    model = SparseAutoModelForCausalLM.from_pretrained(tiny_model_path)

    kwargs = {
        "sparsity": 0.5,
        "block_size": 128,
        "quantize": False,
        "targets": [
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.layers.4",
            "model.layers.5",
        ],
    }

    sparsegpt_modifier_no_head = SparseGPTModifier(
        framework=Framework.pytorch, **kwargs
    )
    state = State(framework=Framework.pytorch)
    state.update(model=model, device=device)
    sparsegpt_modifier_no_head.initialize_compression(state.model)

    kwargs["targets"].append("lm_head")
    sparsegpt_modifier_head = SparseGPTModifier(framework=Framework.pytorch, **kwargs)
    sparsegpt_modifier_head.initialize_compression(state.model)

    # check we pick up the lm_head layer
    layers_no_head = len(sparsegpt_modifier_no_head.compressible_layers_)
    layers_head = len(sparsegpt_modifier_head.compressible_layers_)
    assert layers_head == layers_no_head + 1


def test_sparsities():
    tiny_model_path = "Xenova/llama2.c-stories15M"
    recipe = "tests/sparseml/transformers/obcq/sparse.yaml"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    # test recipe with 50% sparsity, quantization and smoothquant
    oneshot(
        model=tiny_model_path,
        dataset="open_platypus",
        oneshot_device=device,
        recipe=recipe,
        max_seq_length=128,
        num_calibration_samples=64,
        pad_to_max_length=False,
        clear_sparse_session=False,
    )

    model = get_session_model()

    lm_head_sparsity = tensor_sparsity(model.lm_head.weight)
    assert math.isclose(lm_head_sparsity.item(), 0.3, rel_tol=1e-4)
    layer_1_sparse = tensor_sparsity(model.model.layers[1].self_attn.k_proj.weight)
    assert math.isclose(layer_1_sparse.item(), 0.3, rel_tol=1e-4)
    layer_2_dense = tensor_sparsity(model.model.layers[2].self_attn.k_proj.weight)
    assert math.isclose(layer_2_dense.item(), 0.0, rel_tol=1e-4)


def test_sgpt_defaults():
    kwargs = {"sparsity": 0.5}
    sparsegpt_modifier_only_sparsity = SparseGPTModifier(
        framework=Framework.pytorch, **kwargs
    )
    assert not sparsegpt_modifier_only_sparsity.quantize
    assert sparsegpt_modifier_only_sparsity.block_size == 128
    assert sparsegpt_modifier_only_sparsity.sparsity == 0.5

    kwargs = {"quantize": True}
    sparsegpt_modifier_only_quant = SparseGPTModifier(
        framework=Framework.pytorch, **kwargs
    )
    assert sparsegpt_modifier_only_quant.quantize
    assert sparsegpt_modifier_only_quant.block_size == 128
    assert sparsegpt_modifier_only_quant.sparsity == 0.0

    # fail if we don't pass a sparsity or enable quantization
    kwargs = {}
    sparsegpt_invalid = SparseGPTModifier(framework=Framework.pytorch, **kwargs)
    state_test = State(framework=Framework.pytorch)
    sparsegpt_invalid.initialized_structure_ = True
    with pytest.raises(ValueError):
        sparsegpt_invalid.on_initialize(state=state_test)


def test_fake_quant_wrapper(tmp_path):
    from sparseml.transformers import oneshot

    model_name = "roneneldan/TinyStories-1M"
    dataset_name = "open_platypus"
    overwrite_output_dir = True
    precision = "bfloat16"  # unsupported by native FakeQuantize
    oneshot_device = "cuda:0"  # unsupported by native FakeQuantize
    output_dir = tmp_path / "temp_output"
    recipe = """
    first_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - Embedding
                scheme_overrides:
                    LayerNorm:
                        input_activations: null
                        output_activations: null
    """
    num_calibration_samples = 8

    oneshot(
        model=model_name,
        dataset=dataset_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        precision=precision,
        recipe=recipe,
        oneshot_device=oneshot_device,
        num_calibration_samples=num_calibration_samples,
    )


def test_infer_targets():
    model = SparseAutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")
    modifiable_model = ModifiableModel(framework=Framework.pytorch, model=model)
    targets = modifiable_model.get_no_split_params()
    assert len(targets) == 1
    assert targets[0] == "LlamaDecoderLayer"

    modifier = SparseGPTModifierPyTorch(sparsity=0.5)
    modifier.targets = targets
    modifier.model = modifiable_model
    compressible_layers = modifier.compressible_layers()

    # 15M model should have 6 transformer layers
    assert len(compressible_layers) == 6
