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
def mistral_model():
    config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
    with init_empty_weights():
        model = AutoModel.from_config(config)
    return model


def test_modifying_mistral(mistral_model):
    from sparseml.transformers.sparsification.modification.modifying_mistral import (  # noqa F401
        modify,
    )

    mistral_ = deepcopy(mistral_model)
    mistral = modify_model(mistral_model)

    modified_modules_original_model = [
        module
        for module in mistral_.modules()
        if hasattr(module, "attn_output_matmul")
        and module.__class__.__name__ == "MistralAttention"
    ]
    modified_modules_modified_model = [
        module
        for module in mistral.modules()
        if hasattr(module, "attn_output_matmul")
        and module.__class__.__name__ == "MistralAttention"
    ]

    original_modules_original_model = [
        module
        for module in mistral_.modules()
        if not hasattr(module, "attn_output_matmul")
        and module.__class__.__name__ == "MistralAttention"
    ]
    original_modules_modified_model = [
        module
        for module in mistral.modules()
        if not hasattr(module, "attn_output_matmul")
        and module.__class__.__name__ == "MistralAttention"
    ]

    assert (
        len(modified_modules_original_model)
        == len(original_modules_modified_model)
        == 0
    )
    assert (
        len(modified_modules_modified_model)
        == len(original_modules_original_model)
        == 32
    )
