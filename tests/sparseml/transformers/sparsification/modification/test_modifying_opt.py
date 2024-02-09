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
def opt_model():
    config = AutoConfig.from_pretrained("facebook/opt-1.3b")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


def test_modifying_opt(opt_model):
    from sparseml.transformers.sparsification.modification.modifying_opt import (  # noqa F401
        modify,
    )

    num_attn_blocks = opt_model.config.num_hidden_layers

    # keep the original model for comparison
    opt_ = deepcopy(opt_model)
    opt = modify_model(opt_model)

    # check how many modified "OPTAttention" modules are in the original
    # model (should be 0, as the model is not modified yet)
    modified_modules_original_model = [
        module
        for module in opt_.modules()
        if _is_opt_attention_modified(module)
        and module.__class__.__name__ == "OPTAttention"
    ]
    # check how many modified "OPTAttention" modules are
    # in the modified model (should be num_attn_blocks, as the
    # model is modified, and has num_attn_blocks attention blocks)
    modified_modules_modified_model = [
        module
        for module in opt.modules()
        if _is_opt_attention_modified(module)
        and module.__class__.__name__ == "OPTAttention"
    ]
    # check how many original "OPTAttention"
    # modules are in the original
    # model (should be num_attn_blocks, as the model is
    # not modified yet, and has num_attn_blocks attention blocks)
    original_modules_original_model = [
        module
        for module in opt_.modules()
        if not _is_opt_attention_modified(module)
        and module.__class__.__name__ == "OPTAttention"
    ]
    # check how many original "OPTAttention"
    # modules are in the modified
    # model (should be 0, as the model is
    # modified, and should not contain any original
    # "OPTAttention" modules)
    original_modules_modified_model = [
        module
        for module in opt.modules()
        if not _is_opt_attention_modified(module)
        and module.__class__.__name__ == "OPTAttention"
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


def _is_opt_attention_modified(module):
    # only the modified "OPTAttention"
    # modules have the "attn_output_bmm" attribute
    return hasattr(module, "attn_output_bmm")
