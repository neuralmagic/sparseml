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
context of SparseML quantization
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from sparseml.modifiers.quantization_legacy.modification.modification_objects import (
    QuantizableIdentity,
    QuantizableMatMul,
)
from sparseml.modifiers.quantization_legacy.modification.registry import (
    ModificationRegistry,
)
from sparseml.pytorch.utils.helpers import swap_modules
from sparseml.transformers.sparsification.modification.base import (
    check_transformers_version,
)


@ModificationRegistry.register(name="LlamaModel", alias=["LlamaForCausalLM"])
def modify(model: nn.Module) -> nn.Module:
    """
    Modify the LLaMa model to be compatible with SparseML
    quantization

    Replaces the attention modules with
    LlamaAttentionWithQuantizableMatmuls modules

    :param model: the original LLaMa model
    :return: the modified LLaMa model
    """
    check_transformers_version()
    for name, submodule in model.named_modules():
        if isinstance(
            submodule, (LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention)
        ) and not isinstance(submodule, LlamaAttentionWithQuantizableMatmuls):
            swap_modules(model, name, LlamaAttentionWithQuantizableMatmuls(submodule))
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
    Wrapper around the original attention module to introduce
    LlamaAttention with quantizable matmul operations

    :param llama_attention: the original attention module to be
        wrapped and modified
    """

    def __init__(
        self,
        llama_attention: Union[
            LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention
        ],
    ):
        self.__class__ = type(
            self.__class__.__name__,
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
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
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

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache # noqa
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ==== SparseML MODIFICATION ====
        attn_weights = self.attn_weights_matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        # ==============================

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # ==== SparseML MODIFICATION ====
        attn_output = self.attn_output_matmul(attn_weights, value_states)
        # ===============================

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"  # noqa
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
