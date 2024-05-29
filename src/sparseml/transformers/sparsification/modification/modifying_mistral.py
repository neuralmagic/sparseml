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
context of SparseML quantization
"""

import math
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.mistral.modeling_mistral import (
    Cache,
    MistralAttention,
    MistralFlashAttention2,
    MistralSdpaAttention,
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


@ModificationRegistry.register(name="MistralModel", alias=["MistralForCausalLM"])
def modify(model: torch.nn.Module) -> torch.nn.Module:
    """
    Modify the Mistral model to be compatible with SparseML
    quantization

    Replaces the attention modules with
    MistralAttentionWithQuantizableMatmuls modules

    :param model: the original Mistral model
    :return: the modified Mistral model
    """
    check_transformers_version()
    for name, submodule in model.named_modules():
        if isinstance(
            submodule, (MistralAttention, MistralFlashAttention2, MistralSdpaAttention)
        ) and not isinstance(submodule, MistralAttentionWithQuantizableMatmuls):
            swap_modules(model, name, MistralAttentionWithQuantizableMatmuls(submodule))
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
    Wrapper around the original attention module to introduce
    MistralAttention with quantizable matmul operations

    :param mistral_attention: the original attention module to be
        wrapped and modified

    """

    def __init__(
        self,
        mistral_attention: Union[
            MistralAttention, MistralFlashAttention2, MistralSdpaAttention
        ],
    ):
        self.__class__ = type(
            self.__class__.__name__,
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
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"  # noqa
            )
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
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "  # noqa
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "  # noqa
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ==== SparseML MODIFICATION ====
        attn_weights = self.attn_weights_matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        # ===============================

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"  # noqa
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"  # noqa
                )

            attn_weights = attn_weights + attention_mask

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

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
