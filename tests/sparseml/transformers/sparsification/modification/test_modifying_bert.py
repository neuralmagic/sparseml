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
def bert_model():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


def test_modifying_bert(bert_model):
    from sparseml.transformers.sparsification.modification.modifying_bert import (  # noqa F401
        modify,
    )

    num_attn_blocks = bert_model.config.num_hidden_layers

    # keep the original model for comparison
    bert_ = deepcopy(bert_model)
    bert = modify_model(bert_model)

    # check how many modified "BertSelfAttention" modules are in the original
    # model (should be 0, as the model is not modified yet)
    modified_modules_original_model = [
        module
        for module in bert_.modules()
        if _is_bert_attention_modified(module)
        and module.__class__.__name__ == "BertSelfAttention"
    ]
    # check how many modified "BertSelfAttention" modules are
    # in the modified model (should be num_attn_blocks, as the
    # model is modified, and has num_attn_blocks attention blocks)
    modified_modules_modified_model = [
        module
        for module in bert.modules()
        if _is_bert_attention_modified(module)
        and module.__class__.__name__ == "BertSelfAttention"
    ]
    # check how many original "BertSelfAttention"
    # modules are in the original
    # model (should be num_attn_blocks, as the model is
    # not modified yet, and has num_attn_blocks attention blocks)
    original_modules_original_model = [
        module
        for module in bert_.modules()
        if not _is_bert_attention_modified(module)
        and module.__class__.__name__ == "BertSelfAttention"
    ]
    # check how many original "BertSelfAttention"
    # modules are in the modified
    # model (should be 0, as the model is
    # modified, and should not contain any original
    # "BertSelfAttention" modules)
    original_modules_modified_model = [
        module
        for module in bert.modules()
        if not _is_bert_attention_modified(module)
        and module.__class__.__name__ == "BertSelfAttention"
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


def _is_bert_attention_modified(module):
    # only the modified "BertSelfAttention" modules have the
    # modules have the "attention_scores_matmul" attribute
    return hasattr(module, "attention_scores_matmul")
