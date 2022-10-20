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

import torch
import transformers
from transformers.models.mobilebert.modeling_mobilebert import (
    MobileBertEmbeddings as _MobileBertEmbeddings,
)


class QATEmbeddingTransformation(torch.nn.Module):
    def __init__(self, embedded_input_size, hidden_size):
        super().__init__()

        # Behaves like normal Linear module unless a SparseML QuantizationModifier
        # is initialized.
        # When initialized, does not quantize inputs.
        # Only weights are quantized (inputs come quantized from embeddings)
        self.linear = torch.nn.Linear(embedded_input_size, hidden_size)
        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 0,
            "num_outputs": 1,
        }

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class PatchedMobileBertEmbeddings(_MobileBertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = QATEmbeddingTransformation(
            embedded_input_size, config.hidden_size
        )


transformers.models.mobilebert.modeling_mobilebert.MobileBertEmbeddings = (
    PatchedMobileBertEmbeddings
)
