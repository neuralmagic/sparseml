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
Modification to the original LLaMa model required in the
context of SparseML
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
    repeat_kv,
)

from sparseml.pytorch.utils.helpers import swap_modules
from sparseml.transformers.sparsification.modification.modification_objects import (
    QuantizableIdentity,
    QuantizableMatMul,
)
from sparseml.transformers.sparsification.modification.registry import (
    ModificationRegistry,
)


_LOGGER = logging.getLogger(__name__)


@ModificationRegistry.register(name="LlamaModel", alias=["LlamaForCausalLM"])
def modify(model: nn.Module) -> nn.Module:
    """
    Modify the LLaMa model to be compatible with SparseML

    1. Replaces the LlamaAttention modules with
        LlamaAttentionWithQuantizableMatmuls modules

    Note: This function will not alter any of the alternatives
    to the LlamaAttention module such as LlamaFlashAttention2

    :param model: the original LLaMa model
    :return: the modified LLaMa model
    """
    for name, submodule in model.named_modules():
        if isinstance(submodule, LlamaAttention):
            swap_modules(model, name, LlamaAttentionWithQuantizableMatmuls(submodule))
        elif isinstance(submodule, LlamaFlashAttention2):
            _LOGGER.debug(
                f"The model contains {submodule.__class__.__name__} "
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


class MatMulOutput_QK(QuantizableIdentity):
    ...


class MatMulOutput_PV(QuantizableIdentity):
    ...


class LlamaAttentionWithQuantizableMatmuls(LlamaAttention):
    """
    Wrapper around the original LlamaAttention module to replace the
    matmul operations with quantizable matmul operations

    :param llama_attention: the original LlamaAttention module
    """

    def __init__(self, llama_attention: LlamaAttention):
        self.__class__ = type(
            llama_attention.__class__.__name__,
            (self.__class__, llama_attention.__class__),
            {},
        )
        self.__dict__ = llama_attention.__dict__

        self.attn_weights_matmul = QuantizableMatMul(
            MatMulLeftInput_QK, MatMulRightInput_QK, MatMulOutput_QK
        )
        self.attn_output_matmul = QuantizableMatMul(
            MatMulLeftInput_PV, MatMulRightInput_PV, MatMulOutput_PV
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
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

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ==== SparseML MODIFICATION ====
        attn_weights = self.attn_weights_matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        # ==============================

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size "
                    f"{(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # ==== SparseML MODIFICATION ====
        attn_output = self.attn_output_matmul(attn_weights, value_states)
        # ===============================

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
