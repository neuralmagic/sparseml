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
from transformers import AutoConfig, AutoModel

from accelerate import init_empty_weights
from sparseml.transformers.sparsification.modification import modify_model


@pytest.fixture
def mobilebert_model():
    config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


def test_modifying_distilbert(mobilebert_model):
    from sparseml.transformers.sparsification.modification.modifying_distilbert import (  # noqa F401
        modify,
    )

    num_attn_blocks = mobilebert_model.config.num_hidden_layers

    # keep the original model for comparison
    distilbert_ = deepcopy(mobilebert_model)
    distilbert = modify_model(mobilebert_model)

    # check how many modified "MultiHeadSelfAttention" modules are in the original
    # model (should be 0, as the model is not modified yet)
    modified_modules_original_model = [
        module
        for module in distilbert_.modules()
        if _is_distilbert_attention_modified(module)
        and module.__class__.__name__ == "MultiHeadSelfAttention"
    ]
    # check how many modified "MultiHeadSelfAttention" modules are
    # in the modified model (should be num_attn_blocks, as the
    # model is modified, and has num_attn_blocks attention blocks)
    modified_modules_modified_model = [
        module
        for module in distilbert.modules()
        if _is_distilbert_attention_modified(module)
        and module.__class__.__name__ == "MultiHeadSelfAttention"
    ]
    # check how many original "MultiHeadSelfAttention"
    # modules are in the original
    # model (should be num_attn_blocks, as the model is
    # not modified yet, and has num_attn_blocks attention blocks)
    original_modules_original_model = [
        module
        for module in distilbert_.modules()
        if not _is_distilbert_attention_modified(module)
        and module.__class__.__name__ == "MultiHeadSelfAttention"
    ]
    # check how many original "MultiHeadSelfAttention"
    # modules are in the modified
    # model (should be 0, as the model is
    # modified, and should not contain any original
    # "MultiHeadSelfAttention" modules)
    original_modules_modified_model = [
        module
        for module in distilbert.modules()
        if not _is_distilbert_attention_modified(module)
        and module.__class__.__name__ == "MultiHeadSelfAttention"
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


def _is_distilbert_attention_modified(module):
    # only the modified "MultiHeadSelfAttention" modules have the
    # modules have the "attention_scores_matmul" attribute
    return hasattr(module, "attention_scores_matmul")
