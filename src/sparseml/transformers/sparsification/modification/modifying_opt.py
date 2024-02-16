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
Modification to the original OPT model required in the
context of SparseML
"""

import logging
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTAttention

from sparseml.transformers.sparsification.modification.modification_objects import (
    QuantizableBatchMatmul,
    QuantizableIdentity,
    swap_modules,
)
from sparseml.transformers.sparsification.modification.registry import (
    ModificationRegistry,
)


_LOGGER = logging.getLogger(__name__)


@ModificationRegistry.register(name="OPTModel", alias=["OPTForCausalLM"])
def modify(model: nn.Module) -> nn.Module:
    """
    Modify the OPT model to be compatible with SparseML

    1. Replaces the OPTAttention modules with
        OPTAttentionWithQuantizableMatmuls modules

    Note: This function will not alter any of the alternatives
    to the OPTAttention module such as OPTFlashAttention2

    :param model: the original LLaMa model
    :return: the modified LLaMa model
    """
    for name, submodule in model.named_modules():
        submodule_cname = submodule.__class__.__name__
        if submodule_cname == "OPTAttention":
            swap_modules(model, name, OPTAttentionWithQuantizableMatmuls(submodule))
        elif submodule_cname == "OptFlashAttention2":
            _LOGGER.debug(
                f"The model contains {submodule_cname} "
                "module, which will not be modified"
            )
    return model


class BMMLeftInput_QK(QuantizableIdentity):
    ...


class BMMRightInput_QK(QuantizableIdentity):
    ...


class BMMOutput_QK(QuantizableIdentity):
    ...


class BMMLeftInput_PV(QuantizableIdentity):
    ...


class BMMRightInput_PV(QuantizableIdentity):
    ...


class BMMOutput_PV(QuantizableIdentity):
    ...


class OPTAttentionWithQuantizableMatmuls(OPTAttention):
    """
    Wrapper around the original OPTAttention module to replace the
    matmul operations with quantizable matmul operations

    :param opt_attention: the original OPTAttention module
    """

    def __init__(self, opt_attention: OPTAttention):
        self.__class__ = type(
            opt_attention.__class__.__name__,
            (self.__class__, opt_attention.__class__),
            {},
        )
        self.__dict__ = opt_attention.__dict__

        self.attn_weights_bmm = QuantizableBatchMatmul(
            BMMLeftInput_QK, BMMRightInput_QK, BMMOutput_QK
        )
        self.attn_output_bmm = QuantizableBatchMatmul(
            BMMLeftInput_PV, BMMRightInput_PV, BMMOutput_PV
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        This function is almost entirely ported from the
        original OPTAttention module
        (transformers.models.opt.modeling_opt.py::OPTAttention.forward(...)) # noqa: E501
        with the exception of the annotated lines below
        """

        # if key_value_states are provided this layer is
        # used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of
            # all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all
            # cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save
            # Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states.
            # Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to
            # current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value`
            # is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        # ==== SparseML MODIFICATION ====
        attn_weights = self.attn_weights_bmm(query_states, key_states.transpose(1, 2))
        # ==============================
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(
                    torch.finfo(attn_weights.dtype).min, device=attn_weights.device
                ),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16.
        # Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    "Head mask for a single layer "
                    f"should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # ==== SparseML MODIFICATION ====
        attn_output = self.attn_output_bmm(attn_probs, value_states)
        # ==============================

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather
        # than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
