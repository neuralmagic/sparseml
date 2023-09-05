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

from sparseml.pytorch.sparsification.modifier import BaseModifier, PyTorchModifierYAML
from sparseml.pytorch.sparsification.obcq.layer_compressor import LayerCompressor


__all__ = ["SparseGPTModifier"]

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class SparseGPTModifier(BaseModifier):
    def __init__(
        self,
        sparsity: float = 0.5,
        block_size: int = 128,
        quantize: bool = True,
        dampening_frac: Optional[float] = 0.01,
        sequential_update: Optional[bool] = True,
    ):
        super().__init__()

        self._compressible_layers = None
        self._model = None
        self._bottom_compressor = None
        self._head_compressor = None
        self._compressible_layers = None

        self._sparsity = sparsity
        self._block_size = block_size
        self._quantize = quantize
        self._dampening_frac = dampening_frac
        self._sequential_update = sequential_update

        self._device = "cuda:0"

    def compressible_layers(self):
        raise NotImplementedError

    def bottom_compressor(self):
        raise NotImplementedError

    def head_compressor(self):
        raise NotImplementedError

    def one_shot(self, model, dataloader, initializer_kwargs, finalize_kwargs):
        self.initialize(model, **initializer_kwargs)
        extras = self.compress(dataloader)
        finalize_kwargs.update(extras)
        self.finalize(**finalize_kwargs)

    @torch.no_grad()
    def compress(self, dataloader):
        accum_kwargs = {"dataloader": dataloader}
        # Step 0: BottomCompressor accomplishes two things:
        # 1) Compress the embedding if needed
        # 2) Pass the calibration data through the (compressed) bottom part of the
        # network, capturing the outputs
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
            _LOGGER.info(f"\n===== Compressing layer {idx}/{num_layers} =====")
            args = {
                "sparsity": self._sparsity,
                "prunen": 0,
                "prunem": 0,
                "blocksize": self._block_size,
                "percdamp": self._dampening_frac,
                "quantize": self._quantize,
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

        return extras

    def initialize(self, model, **kwargs):
        self.model = model
        self._compressible_layers = self.compressible_layers()
        self._bottom_compressor = self.bottom_compressor()
        self._head_compressor = self.head_compressor()

    def finalize(self, **kwargs):
        use_cache = kwargs.get("use_cache", False)
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache
