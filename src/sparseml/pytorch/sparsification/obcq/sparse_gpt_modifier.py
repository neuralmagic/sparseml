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
from typing import Optional, Union

import torch
from torch import nn

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    BaseModifier,
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.pytorch.sparsification.obcq.layer_compressor import (
    BaseCompressor,
    LayerCompressor,
)
from sparseml.sparsification import SparsificationTypes


__all__ = ["SparseGPTModifier"]

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class SparseGPTModifier(BaseModifier):
    """ """

    def __init__(
        self,
        sparsity: float = 0.5,
        block_size: int = 128,
        quantize: bool = True,
        num_bits: int = 16,
        dampening_frac: Optional[float] = 0.01,
        sequential_update: Optional[bool] = True,
    ):
        super().__init__()

        self._model_preprocessors = []  # filled in by child classes
        self._compressible_layers = None
        self._model = None
        self._bottom_compressor = None
        self._head_compressor = None
        self._compressible_layers = None

        self._sparsity = sparsity
        self._block_size = block_size
        self._quantize = quantize
        self._num_bits = num_bits
        self._dampening_frac = dampening_frac
        self._sequential_update = sequential_update

        self._device = "cuda:0"

    def compressible_layers(self):
        return self.model.model.decoder.layers

    def bottom_compressor(self):
        return OPTBottomCompressor(self.model)

    def head_compressor(self):
        return None  # no head compressor for OPT

    def one_shot(self, model, dataloader, initializer_kwargs, finalize_kwargs):
        self.initialize(model, **initializer_kwargs)
        self.compress(dataloader)
        self.finalize(**finalize_kwargs)

    @torch.no_grad()
    def compress(self, dataloader):
        accum_kwargs = {"dataloader": dataloader}
        # Step 0: BottomCompressor accomplishes two things:
        # 1) Compress the embedding if needed
        # 2) Pass the calibration data through the (compressed) bottom part of the network, capturing the outputs
        # which will become the inputs to the first decoder layer
        # Also return attention_mask as part of kwargs
        self.model, extras = self._bottom_compressor.compress(
            dev=self._device, **accum_kwargs
        )
        accum_kwargs.update(extras)

        # Step 1: Sequentially prune/quantize decoder layers
        inputs = None
        num_layers = len(self._compressible_layers)
        for idx, layer in enumerate(self._compressible_layers):
            if "outputs" not in accum_kwargs:
                raise RuntimeError(
                    "The 'outputs' key is expected but not found from the "
                    "return of the bottom compressor"
                )
            inputs = accum_kwargs["outputs"]
            print(f"\n===== Compressing layer {idx}/{num_layers} =====")
            args = {
                "wbits": self._num_bits,
                "sparsity": self._sparsity,
                "prunen": 0,
                "prunem": 0,
                "blocksize": self._block_size,
                "percdamp": self._dampening_frac,
            }
            layer_compressor = LayerCompressor(
                self.model, layer, idx, inputs, None, args
            )
            # Prune/quantize using SparseGPT
            self.model, layer_kwargs = layer_compressor.compress(
                dev=self._device, **accum_kwargs
            )
            accum_kwargs.update(layer_kwargs)

        # Step 2: Prune/quantize head
        # TODO: Need update here -- see MPT for head quantization example
        if self._head_compressor is not None:
            self.model, extras = self._head_compressor.compress(
                dev=self._device, **accum_kwargs
            )

    def initialize(self, model, **kwargs):
        self.model = model
        self._compressible_layers = self.compressible_layers()
        self._bottom_compressor = self.bottom_compressor()
        self._head_compressor = self.head_compressor()

    def finalize(self, **kwargs):
        use_cache = kwargs.get("use_cache", False)
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache


class OPTBottomCompressor(BaseCompressor):
    """
    OPT specific
    """

    def compress(
        self, dataloader=None, nsamples: int = None, dev: str = "cuda:0", **kwargs
    ):
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

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]

        extras = {
            "use_cache": use_cache,
            "outputs": outs,
            "attention_mask": attention_mask,
        }
        self.model = model
        return model, extras
