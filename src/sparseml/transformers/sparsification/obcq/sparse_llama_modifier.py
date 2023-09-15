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
from torch import nn
from torch.nn import ModuleList

from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.transformers.sparsification.obcq.utils import execute_offloaded_module, cache_attention_inputs
from sparseml.transformers.sparsification.obcq.layer_compressor import BaseCompressor
from sparseml.transformers.sparsification.obcq.sparse_gpt_modifier import (
    SparseGPTModifier,
)


__all__ = ["SparseLlamaModifier"]


class LlamaBottomCompressor(BaseCompressor):
    """
    Llama2 specific
    """

    def compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
        cached_inputs = cache_attention_inputs(self.model, dataloader, dev, nsamples)

        outputs = execute_offloaded_module(
            self.model.model.embed_tokens,
            dataloader,
            dev,
            nsamples,
            overwrite_buffer=False,
        )

        outputs = torch.concatenate(outputs, dim=0)
        cached_inputs.update({"outputs": outputs})
        return self.model, cached_inputs


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
