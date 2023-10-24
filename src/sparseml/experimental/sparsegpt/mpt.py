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
from typing import Dict, Tuple

import torch
import torch.nn as nn

from einops import rearrange
from llmfoundry import (
    COMPOSER_MODEL_REGISTRY,
    build_finetuning_dataloader,
    build_text_denoising_dataloader,
)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.utils.builders import build_tokenizer
from model_preprocessor import ModelPreprocessor, QuantizationModelPreprocessor
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


class SequentialSparseGPT_MPT(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.transformer.blocks


class MPTBottomCompressor(BaseCompressor):
    def compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
        args = kwargs["args"]
        data_seq_len = args.data_sequence_length

        model = self.model
        layers = self.model.model.transformer.blocks

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.transformer.blocks

        model.model.transformer.wte = model.model.transformer.wte.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, data_seq_len, model.config.d_model), dtype=dtype, device=dev
        )
        cache = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[len(cache)] = inp
                cache.append(kwargs["attn_bias"])
                raise ValueError

        layers[0] = Catcher(layers[0])
        i = 0
        for batch in dataloader:
            try:
                tmp = {k: v.to(dev) for k, v in batch.items()}
                model(tmp)
            except ValueError:
                pass
            i += 1
            if i == nsamples:
                break
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.transformer.wte = model.model.transformer.wte.cpu()
        torch.cuda.empty_cache()

        extras = kwargs.copy()
        extras.updates({"use_cache": use_cache, "outputs": inps, "attn_bias": cache})

        self.model = model
        return model, extras


class MPTDecoderLayerCompressor(LayerCompressor):
    ...


class MPTHeadCompressor(BaseCompressor):
    ...


class EmbeddingAndHeadWeightSeparator(ModelPreprocessor):
    """
    Untie embedding and head weights, used for their
    separated quantization
    """

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        from copy import deepcopy

        self.model.model.lm_head = torch.nn.Linear(
            in_features=self.model.model.transformer.wte.weight.shape[1],
            out_features=self.model.model.transformer.wte.weight.shape[0],
            bias=False,
        )
        self.model.model.lm_head.weight = deepcopy(
            self.model.model.transformer.wte.weight
        )
        self.model.model.lm_head = self.model.model.lm_head.to(
            self.model.model.transformer.wte.weight.device
        )

        from typing import List, Optional, Tuple

        import torch.nn.functional as F
        from transformers.modeling_outputs import CausalLMOutputWithPast

        def new_untied_forward(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
        ):
            return_dict = (
                return_dict if return_dict is not None else self.config.return_dict
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            if inputs_embeds is not None:
                raise NotImplementedError(
                    "inputs_embeds has to be None (for hf/peft support)."
                )
            outputs = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                prefix_mask=prefix_mask,
                sequence_id=sequence_id,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
            )

            # [ELDAR] this is from the original implementation
            # logits = self.transformer.wte(
            #     outputs.last_hidden_state.to(self.transformer.wte.weight.device), True
            # )
            # [ELDAR] this is our new version
            logits = self.lm_head(
                outputs.last_hidden_state.to(self.transformer.wte.weight.device)
            )

            if self.logit_scale is not None:
                if self.logit_scale == 0:
                    warnings.warn(
                        f"Multiplying logits by self.logit_scale={self.logit_scale!r}. "
                        "This will produce uniform (uninformative) outputs."
                    )
                logits *= self.logit_scale
            loss = None
            if labels is not None:
                labels = torch.roll(labels, shifts=-1)
                labels[:, -1] = -100
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1)
                )
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # we are overriding forward of an instance, and not a class
        # we should change this to a class method when we own the model implementation
        bound_method = new_untied_forward.__get__(
            self.model.model, self.model.model.__class__
        )
        setattr(self.model.model, "forward", bound_method)


def mpt_get_attn_with_quantized_matmuls(attn_weights_matmul, attn_output_matmul):
    def quantized_scaled_multihead_dot_product_attention(
        query,
        key,
        value,
        n_heads,
        past_key_value=None,
        softmax_scale=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=False,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False,
    ):
        q = rearrange(query, "b s (h d) -> b h s d", h=n_heads)
        kv_n_heads = 1 if multiquery else n_heads
        k = rearrange(key, "b s (h d) -> b h d s", h=kv_n_heads)
        v = rearrange(value, "b s (h d) -> b h s d", h=kv_n_heads)
        if past_key_value is not None:
            if len(past_key_value) != 0:
                k = torch.cat([past_key_value[0], k], dim=3)
                v = torch.cat([past_key_value[1], v], dim=2)
            past_key_value = (k, v)
        (b, _, s_q, d) = q.shape
        s_k = k.size(-1)
        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(d)

        # attn_weight = q.matmul(k) * softmax_scale  # make quantization aware matmul
        attn_weight = (
            attn_weights_matmul(q, k) * softmax_scale
        )  # make quantization aware matmul

        if attn_bias is not None:
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            if (
                attn_bias.size(-1) != 1
                and attn_bias.size(-1) != s_k
                or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q)
            ):
                raise RuntimeError(
                    f"attn_bias (shape: {attn_bias.shape}) is expected to "
                    "broadcast to shape: {attn_weight.shape}."
                )
            attn_weight = attn_weight + attn_bias
        min_val = torch.finfo(q.dtype).min
        if key_padding_mask is not None:
            if attn_bias is not None:
                warnings.warn(
                    "Propogating key_padding_mask to the attention module "
                    + "and applying it within the attention module can cause "
                    + "unneccessary computation/memory usage. Consider integrating "
                    + "into attn_bias once and passing that to each attention "
                    + "module instead."
                )
            attn_weight = attn_weight.masked_fill(
                ~key_padding_mask.view((b, 1, 1, s_k)), min_val
            )
        if is_causal and (not q.size(2) == 1):
            s = max(s_q, s_k)
            causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
            causal_mask = causal_mask.tril()
            causal_mask = causal_mask.to(torch.bool)
            causal_mask = ~causal_mask
            causal_mask = causal_mask[-s_q:, -s_k:]
            attn_weight = attn_weight.masked_fill(
                causal_mask.view(1, 1, s_q, s_k), min_val
            )
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout_p:
            attn_weight = torch.nn.functional.dropout(
                attn_weight, p=dropout_p, training=training, inplace=True
            )
        # out = attn_weight.to(v.dtype).matmul(v)  # make quantization aware matmul
        out = attn_output_matmul(
            attn_weight.to(v.dtype), v
        )  # make quantization aware matmul
        out = rearrange(out, "b h s d -> b s (h d)")
        if needs_weights:
            return (out, attn_weight, past_key_value)
        return (out, None, past_key_value)

    return quantized_scaled_multihead_dot_product_attention


def MatMulQuantizationPreprocessor(ModelPreprocessor):
    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        for name, mod in self.model.named_modules():
            if hasattr(mod, "attn_fn"):
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
                mod.attn_fn = mpt_get_attn_with_quantized_matmuls(
                    mod.attn_weights_matmul, mod.attn_output_matmul
                )
        return self.model, {}


def prepare_sparsegpt(model, dataloader, args, **kwargs) -> SequentialSparseGPT:
    model_preprocessors = []
    if args.recipe:
        # TODO: add check if the recipe has quantiztion modifier
        model_preprocessors.append(MatMulQuantizationPreprocessor(model))
        model_preprocessors.append(EmbeddingAndHeadWeightSeparator(model))
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                model, args.recipe, dataloader, args.observer_batches
            )
        )
    bottom_compressor = MPTBottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_MPT(
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
