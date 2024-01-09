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
from sparseml.transformers.utils.model import SparseCausalLM


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
    dataset_name = "open_platypus"
    if not torch.cuda.is_available():
        device = "cpu"

    # test recipe with 50% sparsity, quantization and smoothquant
    tiny_model = one_shot(
        model_path=tiny_model_path,
        dataset_name=dataset_name,
        num_samples=num_samples,
        device=device,
        recipe_file=recipe_file_path,
    )

    data_args = DataTrainingArguments(
        dataset_name=dataset_name,
        max_seq_length=tiny_model.seqlen,
        num_calibration_samples=num_samples,
        concatenate_data=False,
        pad_to_max_length=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tiny_model_path, use_fast=True, trust_remote_code=True
    )
    dataset_manager = TextGenerationDataset.load_from_registry(
        dataset_name, data_args=data_args, split="train", tokenizer=tokenizer
    )
    raw_dataset = dataset_manager.get_raw_dataset()
    tokenized_dataset = dataset_manager.tokenize_and_process(raw_dataset)
    test_data = format_calibration_data(
        tokenized_dataset=tokenized_dataset, num_calibration_samples=num_samples
    )
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

    model = SparseCausalLM.auto_model_from_pretrained(tiny_model_path)

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
    sparsegpt_modifier_no_head.setup(state)

    kwargs["targets"].append("lm_head")
    sparsegpt_modifier_head = SparseGPTModifier(framework=Framework.pytorch, **kwargs)
    sparsegpt_modifier_head.setup(state)

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
        dataset_name="open_platypus",
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
