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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.nn import Module

from sparseml.pytorch.sparsification.modifier import (
    Modifier,
    ModifierProp,
    PyTorchModifierYAML,
)
from sparseml.transformers.sparsification.obcq.layer_compressor import LayerCompressor


__all__ = ["SparseGPTModifier"]

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class SparseGPTModifier(Modifier):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model. This modifier should
    not be run directly and instead is instantiated from one of the child classes:
    SparseOPTModifier, SparseMPTModifier or SparseLlamaModifier.

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
        quantize: bool = False,
        dampening_frac: Optional[float] = 0.01,
        sequential_update: Optional[bool] = True,
    ):
        super().__init__()

        self._model = None
        self._compressible_layers = None
        self._bottom_compressor = None
        self._head_compressor = None

        self._sparsity = sparsity
        self._block_size = block_size
        self._quantize = quantize
        self._dampening_frac = dampening_frac
        self._sequential_update = sequential_update

        self._device = self._set_device("cuda:0")
        self._finalization_kwargs = {}

    @ModifierProp()
    def sparsity(self) -> float:
        """
        :return: Sparsity to compress model to
        """
        return self._sparsity

    @ModifierProp()
    def block_size(self) -> int:
        """
        :return: Used to determine number of columns to compress in one pass
        """
        return self._block_size

    @ModifierProp()
    def quantize(self) -> bool:
        """
        :return: Whether or not model is quantized (affects layer names)
        """
        return self._quantize

    @ModifierProp()
    def dampening_frac(self) -> float:
        """
        :return: Amount of dampening to apply to H, as a fraction of the diagonal norm
        """
        return self._dampening_frac

    @ModifierProp()
    def sequential_update(self) -> bool:
        """
        :return: Whether or not to update weights sequentially by layer, True saves on
            GPU memory
        """
        return self._sequential_update

    def compressible_layers(self):
        raise NotImplementedError  # must be implemented by child class

    def bottom_compressor(self):
        raise NotImplementedError  # must be implemented by child class

    def head_compressor(self):
        raise NotImplementedError  # must should be implemented by child class

    def initialize(
        self,
        model: Module,
        calibration_dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
        device: Optional[str] = "cuda:0",
    ):
        """
        Initializes the compressible layers of model(architecture-specific), sets the
        device and runs sparsification on model

        :param model: PyTorch model to sparsify
        :param calibration_dataloader: data to use for calibration during sparsification
        :param device: device to run sparsification on, preferably a GPU
        """
        self.model = model
        self._compressible_layers = self.compressible_layers()
        self._bottom_compressor = self.bottom_compressor()
        self._head_compressor = self.head_compressor()
        self._set_device(device)

        extras = self.compress(calibration_dataloader)
        self._finalization_kwargs.update(extras)

    def finalize(self, model: Module):
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param model: un-used, for matching spec of other modifiers
        """
        use_cache = self._finalization_kwargs.get("use_cache", False)
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

    @torch.no_grad()
    def compress(self, dataloader) -> Dict:
        """
        Run OBCQ on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for OBCQ
        :return: compression outputs used for finalization
        """
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
                "sequential_update": self._sequential_update,
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
        if self._head_compressor is not None:
            self.model, extras = self._head_compressor.compress(
                dev=self._device, **accum_kwargs
            )

        return extras

    def _set_device(self, device: str):
        if "cuda" in device and not torch.cuda.is_available():
            self._device = "cpu"
        else:
            self._device = device
