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
from torch.nn import ModuleList

from sparseml.modifiers.sparsification.obcq.base import (
    SparseGPTModifier,
    SparseLlamaModifier,
    SparseOPTModifier,
)
from sparseml.modifiers.sparsification.obcq.utils.bottom_compressors import (
    LlamaBottomCompressor,
    OPTBottomCompressor,
)
from sparseml.modifiers.sparsification.obcq.utils.layer_compressor import (
    LayerCompressor,
)


_LOGGER = logging.getLogger(__name__)


class SparseGPTModifierPyTorch(SparseGPTModifier):
    model: Any = None
    compressible_layers_: Any = None
    bottom_compressor_: Any = None
    head_compressor_: Any = None
    device_: str = "cuda:0"
    finalization_kwargs_: Dict = None

    def compressible_layers(self):
        raise NotImplementedError  # must be implemented by child class

    def bottom_compressor(self):
        raise NotImplementedError  # must be implemented by child class

    def head_compressor(self):
        raise NotImplementedError  # must should be implemented by child class

    def on_initialize(self, state: "State", **kwargs) -> bool:
        self.finalization_kwargs_ = {}
        model = state.model.model
        calibration_dataloader = state.data.calib
        device = state.hardware.device

        self.initialize_obcq(model, device)
        extras = self.apply_obcq(calibration_dataloader)
        self.finalization_kwargs_.update(extras)

        return True

    def initialize_obcq(
        self,
        model: "Module",
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
        self.compressible_layers_ = self.compressible_layers()
        self.bottom_compressor_ = self.bottom_compressor()
        self.head_compressor_ = self.head_compressor()
        self._set_device(device)

    @torch.no_grad()
    def apply_obcq(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
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
        self.model, extras = self.bottom_compressor_.compress(
            dev=self.device_, **accum_kwargs
        )
        accum_kwargs.update(extras)

        # Step 1: Sequentially prune/quantize decoder layers
        inputs = None
        num_layers = len(self.compressible_layers_)
        for idx, layer in enumerate(self.compressible_layers_):
            if "outputs" not in accum_kwargs:
                raise RuntimeError(
                    "The 'outputs' key is expected but not found from the "
                    "return of the bottom compressor"
                )
            inputs = accum_kwargs["outputs"]
            _LOGGER.info(f"\n===== Compressing layer {idx}/{num_layers} =====")
            args = {
                "sparsity": self.sparsity,
                "prunen": 0,
                "prunem": 0,
                "blocksize": self.block_size,
                "percdamp": self.dampening_frac,
                "sequential_update": self.sequential_update,
                "quantize": self.quantize,
            }
            layer_compressor = LayerCompressor(
                self.model, layer, idx, inputs, None, args
            )
            # Prune/quantize using SparseGPT
            self.model, layer_kwargs = layer_compressor.compress(
                dev=self.device_, **accum_kwargs
            )
            accum_kwargs.update(layer_kwargs)

        # Step 2: Prune/quantize head
        if self.head_compressor_ is not None:
            self.model, extras = self.head_compressor_.compress(
                dev=self.device_, **accum_kwargs
            )

        return extras

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param model: un-used, for matching spec of other modifiers
        """
        use_cache = self.finalization_kwargs_.get("use_cache", False)
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

        return True

    def _set_device(self, device: str):
        if "cuda" in device and not torch.cuda.is_available():
            self.device_ = "cpu"
        else:
            self.device_ = device


class SparseLlamaModifierPyTorch(SparseGPTModifierPyTorch, SparseLlamaModifier):
    def compressible_layers(self) -> ModuleList:
        return self.model.model.layers

    def bottom_compressor(self) -> LlamaBottomCompressor:
        return LlamaBottomCompressor(self.model)

    def head_compressor(self) -> None:
        return None  # no head compressor for OPT


class SparseOPTModifierPyTorch(SparseGPTModifierPyTorch, SparseOPTModifier):
    """
    OPT-specific functions for applying the one-shot OBCQ algorithm to a model
    """

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
