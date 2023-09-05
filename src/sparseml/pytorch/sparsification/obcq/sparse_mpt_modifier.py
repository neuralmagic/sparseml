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

import logging
from typing import Optional

import torch
from torch import nn

from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.pytorch.sparsification.obcq.layer_compressor import BaseCompressor
from sparseml.pytorch.sparsification.obcq.sparse_gpt_modifier import SparseGPTModifier


_LOGGER = logging.getLogger(__name__)

__all__ = ["SparseMPTModifier"]


class MPTBottomCompressor(BaseCompressor):
    def compress(self, dev: str = "cuda:0", **kwargs):
        NSAMPLES = kwargs["nsamples"]
        data_seq_len = kwargs["data_seq_len"]
        dataloader = kwargs["dataloader"]

        model = self.model
        layers = self.model.model.transformer.blocks

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.transformer.blocks

        model.model.transformer.wte = model.model.transformer.wte.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (NSAMPLES, data_seq_len, model.config.d_model), dtype=dtype, device=dev
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
            if i == NSAMPLES:
                break
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.transformer.wte = model.model.transformer.wte.cpu()
        torch.cuda.empty_cache()

        extras = kwargs.copy()
        extras.updates({"use_cache": use_cache, "outputs": inps, "attn_bias": cache})

        self.model = model
        return model, extras


class SparseMPTModifier(SparseGPTModifier):
    """ """

    def __init__(
        self,
        sparsity: float = 0.5,
        block_size: int = 128,
        quantize: bool = True,
        num_bits: int = 16,
        dampening_frac: Optional[float] = 0.001,
        sequential_update: Optional[bool] = True,
    ):
        super().__init__(
            sparsity=sparsity,
            block_size=block_size,
            quantize=quantize,
            num_bits=num_bits,
            dampening_frac=dampening_frac,
            sequential_update=sequential_update,
        )

    def compressible_layers(self):
        return self.model.model.transformer.blocks

    def bottom_compressor(self):
        return MPTBottomCompressor(self.model)

    def head_compressor(self):
        return None  # no head compressor for MPT
