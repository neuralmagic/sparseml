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

"""
Modification to the original DistilBert model required in the
context of SparseML quantization
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertFlashAttention2,
    MultiHeadSelfAttention,
)

from sparseml.modifiers.quantization.modification.modification_objects import QATMatMul
from sparseml.modifiers.quantization.modification.registry import ModificationRegistry
from sparseml.pytorch.utils.helpers import swap_modules
from sparseml.transformers.sparsification.modification.base import (
    check_transformers_version,
)


@ModificationRegistry.register(name="DistilBertModel")
def modify(model: nn.Module) -> nn.Module:
    """
    Modify the DistilBert model to be compatible with SparseML
    quantization

    Replaces the attention modules with
    MultiHeadSelfAttentionWithQuantizableMatmuls modules

    :param model: the original DistilBert model
    :return: the modified DistilBert model
    """
    check_transformers_version()
    for name, submodule in model.named_modules():
        if isinstance(
            submodule, (MultiHeadSelfAttention, DistilBertFlashAttention2)
        ) and not isinstance(submodule, MultiHeadSelfAttentionWithQuantizableMatmuls):
            swap_modules(
                model, name, MultiHeadSelfAttentionWithQuantizableMatmuls(submodule)
            )
    return model


class MultiHeadSelfAttentionWithQuantizableMatmuls(MultiHeadSelfAttention):
    """
    Wrapper around the original attention module to introduce
    MultiHeadSelfAttention  with quantizable matmul operations

    :param mhs_attention: the original attention module to be
        wrapped and modified
    """

    def __init__(self, mhs_attention: MultiHeadSelfAttention):
        self.__class__ = type(
            self.__class__.__name__,
            (self.__class__, mhs_attention.__class__),
            {},
        )
        self.__dict__ = mhs_attention.__dict__

        self.attention_scores_matmul = QATMatMul
        self.context_layer_matmul = QATMatMul

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs, # noqa
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True` # noqa
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured' # noqa
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        # ==== SparseML MODIFICATION ====
        scores = self.attention_scores_matmul(
            q, k.transpose(2, 3)
        )  # (bs, n_heads, q_length, k_length)
        # ===============================
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(
            scores, dim=-1
        )  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        # ==== SparseML MODIFICATION ====
        context = self.context_layer_matmul(
            weights, v
        )  # (bs, n_heads, q_length, dim_per_head)
        # ===============================
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)
