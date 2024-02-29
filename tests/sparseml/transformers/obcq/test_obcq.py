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
from transformers import AutoTokenizer

from sparseml.core.framework import Framework
from sparseml.core.state import State
from sparseml.modifiers.obcq import SparseGPTModifier
from sparseml.modifiers.obcq.utils.helpers import ppl_eval_general
from sparseml.pytorch.utils.helpers import tensor_sparsity
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import format_calibration_data
from sparseml.transformers.sparsification.obcq.obcq import one_shot
from sparseml.transformers.sparsification.obcq.utils.helpers import llama_forward
from sparseml.transformers.utils.helpers import resolve_sequence_length
from sparseml.transformers.utils.initializers import (
    initialize_config,
    initialize_sparse_model,
)


@pytest.mark.parametrize(
    "recipe_file_path",
    [
        "tests/sparseml/transformers/obcq/test_tiny.yaml",
        "tests/sparseml/transformers/obcq/test_tiny2.yaml",
        "tests/sparseml/transformers/obcq/test_tiny_w_head.yaml",
    ],
)
def test_obcq_tinystories(recipe_file_path):
    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    num_samples = 64
    dataset = "open_platypus"
    if not torch.cuda.is_available():
        device = "cpu"
    config = initialize_config(model_path=tiny_model_path)

    # test recipe with 50% sparsity, quantization and smoothquant
    tiny_model = one_shot(
        model_path=tiny_model_path,
        dataset=dataset,
        num_samples=num_samples,
        device=device,
        recipe_file=recipe_file_path,
    )

    data_args = DataTrainingArguments(
        dataset=dataset,
        max_seq_length=resolve_sequence_length(config),
        num_calibration_samples=num_samples,
        concatenate_data=False,
        pad_to_max_length=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tiny_model_path, use_fast=True, trust_remote_code=True
    )
    dataset_manager = TextGenerationDataset.load_from_registry(
        dataset, data_args=data_args, split="train", tokenizer=tokenizer
    )
    raw_dataset = dataset_manager.get_raw_dataset()
    tokenized_dataset = dataset_manager.tokenize_and_process(raw_dataset)
    test_data = format_calibration_data(
        tokenized_dataset=tokenized_dataset, num_calibration_samples=num_samples
    )
    test_data = [d["input_ids"] for d in test_data]
    perplexity = ppl_eval_general(
        llama_forward, tiny_model, test_data, device, max_samples_per_iteration=8
    )

    # we aren't expecting good results from this tiny model, but this should catch any
    # egregious errors with the OBCQ algorithm
    assert perplexity < 10000.0


def test_lm_head_target():
    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    config = initialize_config(model_path=tiny_model_path)
    model = initialize_sparse_model(
        model_path=tiny_model_path,
        device=device,
        task="text-generation",
        config=config,
    )

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
    lm_head_recipe = "tests/sparseml/transformers/obcq/test_tiny_w_head.yaml"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    # test recipe with 50% sparsity, quantization and smoothquant
    tiny_model = one_shot(
        model_path=tiny_model_path,
        dataset="open_platypus",
        num_samples=64,
        device=device,
        recipe_file=lm_head_recipe,
    )

    lm_head_sparsity = tensor_sparsity(tiny_model.lm_head.weight)
    assert math.isclose(lm_head_sparsity.item(), 0.3, rel_tol=1e-4)
    layer_1_sparse = tensor_sparsity(tiny_model.model.layers[1].self_attn.k_proj.weight)
    assert math.isclose(layer_1_sparse.item(), 0.3, rel_tol=1e-4)
    layer_2_dense = tensor_sparsity(tiny_model.model.layers[2].self_attn.k_proj.weight)
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
