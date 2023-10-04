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

from sparseml.core.model import ModifiableModel
from sparseml.core.state import State
from sparseml.modifiers.obcq.base import SparseGPTModifier
from sparseml.modifiers.obcq.utils.layer_compressor import (
    BaseCompressor,
    LayerCompressor,
)


_LOGGER = logging.getLogger(__name__)


class SparseGPTModifierPyTorch(SparseGPTModifier):
    """
    Pytorch implementation of SparseGPT

    Lifecycle:
        - on_initialize
            - initialize_obcq
            - apply_obcq
        - on_finalize

    :param model: Pytorch model to perform OBCQ on, in-place
    """

    model: Module = None
    compressible_layers_: List[Module] = None
    bottom_compressor_: BaseCompressor = None
    device_: str = "cuda:0"
    finalization_kwargs_: Dict = None

    def compressible_layers(self) -> List[Module]:
        """
        Retrieves the modules corresponding to a list of compressible layer names

        :return: list of Pytorch modules to compress
        """
        compressible_dict = self.model.get_layers(self.compress_layers)
        return [v for _, v in compressible_dict.items()]

    def bottom_compressor(self) -> BaseCompressor:
        """
        Get a bottom compressor for the module, which runs everything up to the first
        decoder layer

        :return: compressor for bottom part of the network, feeds into first decoder
        """
        return BaseCompressor(self.model)

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        self.finalization_kwargs_ = {}
        modifiable_model = state.model
        calibration_dataloader = state.data.calib
        device = state.hardware.device

        self.initialize_obcq(modifiable_model, device)
        extras = self.apply_obcq(calibration_dataloader)
        self.finalization_kwargs_.update(extras)

        return True

    def initialize_obcq(
        self,
        model: "ModifiableModel",
        device: Optional[str] = "cuda:0",
    ):
        """
        Setup for SparseGPT, initialize the the compressible layers of model, set the
        device, and initialize the bottom compressor

        :param model: PyTorch model to sparsify
        :param device: device to run sparsification on, preferably a GPU
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.model = self.model.model
        self.bottom_compressor_ = self.bottom_compressor()
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
            dev=self.device_,
            target_ids=self.target_ids,
            layer_prefix=self.layer_prefix,
            **accum_kwargs,
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
                "prunen": self.prunen,
                "prunem": self.prunem,
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

        return extras

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param state: un-used, for matching spec of Modifier base class
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
