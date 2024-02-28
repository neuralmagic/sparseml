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

from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention


def test_modifying_distilbert(distilbert_model, helpers):
    from sparseml.transformers.sparsification.modification.modifying_distilbert import (  # noqa F401
        modify,
    )

    helpers.check_model_modified(
        distilbert_model,
        module_to_replace=MultiHeadSelfAttention,
        func_to_validate_replacement=_is_distilbert_attention_modified,
    )


def _is_distilbert_attention_modified(module):
    # only the modified "MultiHeadSelfAttention" modules have the
    # modules have the "attention_scores_matmul" attribute
    return hasattr(module, "attention_scores_matmul")
