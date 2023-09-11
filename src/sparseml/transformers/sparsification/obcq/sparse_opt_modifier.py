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
from sparseml.transformers.sparsification.obcq.layer_compressor import BaseCompressor
from sparseml.transformers.sparsification.obcq.sparse_gpt_modifier import (
    SparseGPTModifier,
)


__all__ = ["SparseOPTModifier"]


class OPTBottomCompressor(BaseCompressor):
    """
    The OPT-specific BottomCompressor accomplishes three things:
        1) Compress the embedding if needed
        2) Pass the calibration data through the (compressed) bottom part of the
        network, capturing the outputs which will become the inputs to the first
        decoder layer
        3) Return attention_mask as part of kwargs
    """

    def compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
        """
        :param dataloader: calibration data to pass through the model
        :nsamples: number of samples to use for calibration, or None to use it all
        :dev: device to use
        :return: model used for calibration, outputs from bottom part of network,
        attention mask, and kv-cache state
        """
        model = self.model
        layers = model.model.decoder.layers
        nsamples = len(dataloader) if nsamples is None else nsamples

        use_cache = model.config.use_cache
        model.config.use_cache = False

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"i": 0, "attention_mask": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()

        attention_mask = cache["attention_mask"]

        extras = {
            "use_cache": use_cache,
            "outputs": inps,
            "attention_mask": attention_mask,
        }
        self.model = model
        return model, extras


@PyTorchModifierYAML()
class SparseOPTModifier(SparseGPTModifier):
    """
    OPT-specific functions for applying the one-shot OBCQ algorithm to a model

    Life-cycle:
        - initialze
            - compress
        - finalize

    :param sparsity: Sparsity to compress model to
    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Whether or not model is quantized (affects layer names)
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    """

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
        """
        :return: list of OPT submodules that can be sparsified
        """
        return self.model.model.decoder.layers

    def bottom_compressor(self) -> OPTBottomCompressor:
        """
        :return: model used for calibration, outputs from bottom part of network,
        attention mask, and kv-cache state
        """
        return OPTBottomCompressor(self.model)

    def head_compressor(self) -> None:
        return None  # no head compressor for OPT
