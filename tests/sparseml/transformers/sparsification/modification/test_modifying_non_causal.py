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

from sparseml.transformers.sparsification.modification.modifying_bert import (
    BertSelfAttentionWithQuantizableMatmuls,
)
from sparseml.transformers.sparsification.modification.modifying_distilbert import (
    MultiHeadSelfAttentionWithQuantizableMatmuls,
)
from sparseml.transformers.sparsification.modification.modifying_mobilebert import (
    MobileBertEmbeddingsWithQuantizableLinear,
)


def test_modify_distilbert(distilbert_model, shared_helper_functions):
    shared_helper_functions.check_model_modified_non_causal(
        distilbert_model, modified_module=MultiHeadSelfAttentionWithQuantizableMatmuls
    )


def test_modify_bert(bert_model, shared_helper_functions):
    shared_helper_functions.check_model_modified_non_causal(
        bert_model, modified_module=BertSelfAttentionWithQuantizableMatmuls
    )


def test_modify_mobilebert(mobilebert_model, shared_helper_functions):
    shared_helper_functions.check_model_modified_non_causal(
        mobilebert_model,
        modified_module=MobileBertEmbeddingsWithQuantizableLinear,
        num_modified_modules=1,
    )
