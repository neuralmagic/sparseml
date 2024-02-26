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
Modification to the original Mistral model required in the
context of SparseML
"""
import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from sparseml.transformers.sparsification.modification.modification_objects import (
    QuantizableIdentity,
    QuantizableMatMul,
    swap_modules,
)
from sparseml.transformers.sparsification.modification.registry import (
    ModificationRegistry,
)


_LOGGER = logging.getLogger(__name__)


@ModificationRegistry.register(name="MistralModel")
def modify(model: torch.nn.Module) -> torch.nn.Module:
    """
    Modify the Mistral model to be compatible with SparseML

    1. Replaces the MistralAttention modules with
        MistralAttentionWithQuantizableMatmuls modules

    :param model: the original Mistral model
    :return: the modified Mistral model
    """
    for name, submodule in model.named_modules():
        submodule_cname = submodule.__class__.__name__
        if submodule_cname == "MistralAttention":
            swap_modules(model, name, MistralAttentionWithQuantizableMatmuls(submodule))
        if submodule_cname == "MistralFlashAttention2":
            _LOGGER.debug(
                f"The model contains {submodule_cname} "
                "module, which will not be modified"
            )
    return model


class MatMulLeftInput_QK(QuantizableIdentity):
    ...


class MatMulRightInput_QK(QuantizableIdentity):
    ...


class MatMulLeftInput_PV(QuantizableIdentity):
    ...


class MatMulRightInput_PV(QuantizableIdentity):
    ...


class MistralAttentionWithQuantizableMatmuls(MistralAttention):
    """
    Wrapper around the original MistralAttention module to replace the
    matmul operations with quantizable matmul operations

    :param mistral_attention: the original MistralAttention module

    """

    def __init__(self, mistral_attention: MistralAttention):
        self.__class__ = type(
            mistral_attention.__class__.__name__,
            (self.__class__, mistral_attention.__class__),
            {},
        )
        self.__dict__ = mistral_attention.__dict__

        self.attn_weights_matmul = QuantizableMatMul(
            MatMulLeftInput_QK, MatMulRightInput_QK
        )
        self.attn_output_matmul = QuantizableMatMul(
            MatMulLeftInput_PV, MatMulRightInput_PV
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        This function is almost entirely ported from the
        original MistralAttention module
        (transformers.models.mistral.modeling_mistral.py::MistralAttention.forward(...)) # noqa: E501
        with the exception of the annotated lines below
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ==== SparseML MODIFICATION ====
        attn_weights = self.attn_weights_matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        # ======================
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"  # noqa: E501
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"  # noqa: E501
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # ==== SparseML MODIFICATION ====
        attn_output = self.attn_output_matmul(attn_weights, value_states)
        # ======================
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"  # noqa: E501
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
