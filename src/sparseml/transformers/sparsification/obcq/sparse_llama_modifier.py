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

from typing import Optional

import torch
from torch.nn import ModuleList

from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.transformers.sparsification.obcq.layer_compressor import BaseCompressor
from sparseml.transformers.sparsification.obcq.sparse_gpt_modifier import (
    SparseGPTModifier,
)
from sparseml.transformers.sparsification.obcq.utils import (
    catch,
    execute_offloaded_module,
)


__all__ = ["SparseLlamaModifier"]


class LlamaBottomCompressor(BaseCompressor):
    """
    Llama2 specific
    """

    @staticmethod
    def _cache_attention_inputs(model, dataloader, device, nsamples):
        model.model.embed_tokens.to(device)
        model.model.layers[0].to(device)
        cached_inputs = catch(
            model,
            model.model.layers[0],
            ["attention_mask", "position_ids"],
            dataloader,
            nsamples,
        )
        model.model.embed_tokens.cpu()
        model.model.layers[0].cpu()
        torch.cuda.empty_cache()
        return cached_inputs

    @staticmethod
    def forward(model, data_loader, device, nsamples=None):
        # Catch attention mask
        cached_inputs = LlamaBottomCompressor._cache_attention_inputs(
            model, data_loader, device, nsamples
        )
        buffer = [b[0] for b in cached_inputs.pop("inputs")]
        for layer in model.model.layers:
            buffer = execute_offloaded_module(
                layer,
                buffer,
                device,
                cached_inputs=cached_inputs,
                use_cache=False,
            )
            buffer = [b[0] for b in buffer]

        del cached_inputs
        torch.cuda.empty_cache()

        buffer = execute_offloaded_module(
            model.model.norm,
            buffer,
            device,
        )
        logits = execute_offloaded_module(
            model.lm_head,
            buffer,
            device,
        )

        return logits


@PyTorchModifierYAML()
class SparseLlamaModifier(SparseGPTModifier):
    def __init__(
        self,
        sparsity: float = 0.5,
        block_size: int = 128,
        quantize: bool = True,
        dampening_frac: Optional[float] = 0.01,
        sequential_update: Optional[bool] = True,
    ):
        super().__init__(
            sparsity=sparsity,
            block_size=block_size,
            quantize=quantize,
            dampening_frac=dampening_frac,
            sequential_update=sequential_update,
        )

    def compressible_layers(self) -> ModuleList:
        return self.model.model.layers

    def bottom_compressor(self) -> LlamaBottomCompressor:
        return LlamaBottomCompressor(self.model)

    def head_compressor(self) -> None:
        return None  # no head compressor for OPT
