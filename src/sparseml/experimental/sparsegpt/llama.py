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

import contextlib
import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from llmfoundry import (
    COMPOSER_MODEL_REGISTRY,
    build_finetuning_dataloader,
    build_text_denoising_dataloader,
)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.utils.builders import build_tokenizer
from model_preprocessor import QuantizationModelPreprocessor
from omegaconf import OmegaConf as om
from sparseml.experimental.sparsegpt.layer_compressor import (
    BaseCompressor,
    LayerCompressor,
)
from sparseml.experimental.sparsegpt.quant import (
    MatMulLeftInput_PV,
    MatMulLeftInput_QK,
    MatMulOutput_PV,
    MatMulOutput_QK,
    MatMulRightInput_PV,
    MatMulRightInput_QK,
    QuantizableMatMul,
)
from sparseml.experimental.sparsegpt.sequential import SequentialSparseGPT


class SequentialSparseGPT_LLAMA(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.model.layers


class LLAMABottomCompressor(BaseCompressor):
    def compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
        args = kwargs["args"]
        data_seq_len = args.data_sequence_length

        model = self.model
        layers = self.model.model.transformer.blocks

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.model.layers

        model.model.model.embed_tokens = model.model.model.embed_tokens.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, data_seq_len, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        i = 0
        for batch in dataloader:
            try:
                tmp = {k: v.to(dev) for k, v in batch.items()}
                # cache_attn_mask.append(tmp["attention_mask"])
                model(tmp)
            except ValueError:
                pass
            i += 1
            if i == nsamples:
                break
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        extras = {
            "use_cache": use_cache,
            "outputs": outs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        self.model = model
        return model, extras


class LLAMADecoderLayerCompressor(LayerCompressor):
    ...


class LLAMAHeadCompressor(BaseCompressor):
    ...


def llama2_get_attn_with_quantized_matmuls(attn_weights_matmul, attn_output_matmul):
    def forward_with_quantized_bmms(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = attn_weights_matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

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
        attn_output = attn_output_matmul(attn_weights, value_states)

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

    return forward_with_quantized_bmms


def MatMulQuantizationPreprocessor(ModelPreprocessor):
    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        for name, mod in self.model.named_modules():
            if isinstance(mod, LlamaAttention):
                print(
                    f"Overriding attention for {name} with quantization-aware matmuls"
                )
                attn_weights_matmul = QuantizableMatMul(
                    MatMulLeftInput_QK, MatMulRightInput_QK, MatMulOutput_QK
                )
                attn_output_matmul = QuantizableMatMul(
                    MatMulLeftInput_PV, MatMulRightInput_PV, MatMulOutput_PV
                )
                mod.attn_weights_matmul = attn_weights_matmul
                mod.attn_output_matmul = attn_output_matmul

                # we are overriding forward of an instance, and not a class
                # we should change this to a class method when we own the
                # model implementation
                bound_method = llama2_get_attn_with_quantized_matmuls(
                    mod.attn_weights_matmul, mod.attn_output_matmul
                ).__get__(mod, mod.__class__)
                setattr(mod, "forward", bound_method)
        return self.model, {}


def prepare_sparsegpt(model, dataloader, args, **kwargs) -> SequentialSparseGPT:
    # TODO: Check with Eldar on additional preprocessing (e.g., weight untying)
    model_preprocessors = []
    if args.recipe:
        model_preprocessors.append(MatMulQuantizationPreprocessor(model))
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                args.recipe, dataloader, args.observer_batches
            )
        )
    bottom_compressor = LLAMABottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_LLAMA(
        model,
        recipe=args.recipe,
        model_preprocessors=model_preprocessors,
        bottom_compressor=bottom_compressor,
    )

    return sequential_sparsegpt


def load_model(args):
    cfg = _build_cfg(args)
    tokenizer = build_tokenizer(cfg.tokenizer)

    print("Initializing model...")
    init_context = contextlib.nullcontext()
    cfg.model.init_device = "cpu"
    with init_context:
        model = build_composer_model(cfg.model, tokenizer)
    return model, {"cfg": cfg, "tokenizer": tokenizer}


def load_data(args):
    cfg = _build_cfg(args)
    tokenizer = build_tokenizer(cfg.tokenizer)
    train_loader = build_dataloader(
        cfg.train_loader,
        tokenizer,
        cfg.device_train_batch_size,
    )
    test_loader = build_dataloader(
        cfg.eval_loader, tokenizer, cfg.device_eval_batch_size
    )

    return train_loader, test_loader, tokenizer


def build_composer_model(model_cfg, tokenizer):
    warnings.filterwarnings(
        action="ignore",
        message="Torchmetrics v0.9 introduced a new argument class property",
    )
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_cfg.name}")
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)


def _build_cfg(args):
    yaml_path = args.yaml_path
    args_list = args.args_list

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    return cfg


def build_dataloader(cfg, tokenizer, device_batch_size):
    if cfg.name == "text":
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "text_denoising":
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "finetuning":
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    else:
        raise ValueError(f"Not sure how to build dataloader with config: {cfg}")
