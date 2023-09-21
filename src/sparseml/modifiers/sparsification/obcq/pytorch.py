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

from sparseml.modifiers.sparsification.obcq.base import SparseGPTModifier


_LOGGER = logging.getLogger(__name__)


class SparseGPTModifierPyTorch(SparseGPTModifier):
    model_: "Module" = None
    compressible_layers_: "ModuleList" = None
    bottom_compressor_: "BaseCompressor" = None
    head_compressor_: "BaseCompressor" = None
    device_: str = "cuda:0"
    finalization_kwargs_: Dict = None

    def compressible_layers(self):
        raise NotImplementedError  # must be implemented by child class

    def bottom_compressor(self):
        raise NotImplementedError  # must be implemented by child class

    def head_compressor(self):
        raise NotImplementedError  # must should be implemented by child class

    def on_initialize(self, state: "State", **kwargs) -> bool:
        model = state.model.model
        calibration_dataloader = state.data.calib.data
        device = state.hardware.device

        self.initialize_obcq(model, calibration_dataloader, device)
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
            dev=self._device, **accum_kwargs
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
        if self.head_compressor_ is not None:
            self.model, extras = self.head_compressor_.compress(
                dev=self._device, **accum_kwargs
            )

        return extras

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param model: un-used, for matching spec of other modifiers
        """
        use_cache = self._finalization_kwargs.get("use_cache", False)
        self.model.apply(torch.quantization.disable_observer)
        self.model.config.use_cache = use_cache

    def _set_device(self, device: str):
        if "cuda" in device and not torch.cuda.is_available():
            self._device = "cpu"
        else:
            self._device = device
