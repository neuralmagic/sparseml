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
import torch.nn as nn


class TextModel(nn.Module):
    def __init__(
        self,
        token_embedding: torch.nn.Embedding,
        positional_embedding: torch.nn.parameter.Parameter,
        transformer: torch.nn.Module,
        ln_final: torch.nn.LayerNorm,
        text_projection: torch.nn.parameter.Parameter,
        attn_mask: torch.Tensor,
    ):

        super().__init__()

        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection
        self.attn_mask = attn_mask
        self.cast_dtype = self.transformer.get_cast_dtype()

    def forward(self, input_ids):
        x = self.token_embedding(input_ids).to(self.cast_dtype)
        x = x + self.positional_embedding.to(self.cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token = highest in each sequence)
        x = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection
        return x
