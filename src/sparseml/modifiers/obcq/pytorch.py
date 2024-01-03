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

from sparseml.core.model import ModifiableModel
from sparseml.core.state import State
from sparseml.modifiers.obcq.base import SparseGPTModifier
from sparseml.modifiers.obcq.utils.layer_compressor import LayerCompressor
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward


_LOGGER = logging.getLogger(__name__)


class SparseGPTModifierPyTorch(SparseGPTModifier):
    """
    Pytorch implementation of SparseGPT

    Lifecycle:
        - on_initialize
            - initialize_obcq
                - compressible_layers
            - apply_obcq
                - compress_bottom
                - LayerCompressor.compress
        - on_finalize

    :param model: Pytorch model to perform OBCQ on, in-place
    """

    model: Any = None
    layer_compressors: List = None
    layer_prefix_: Optional[str] = None

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        self._validate_layerwise_sparsity()

        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)
        if self.quantization_modifier_:
            self.quantization_modifier_.initialize(state, **kwargs)
        modifiable_model = state.model
        calibration_dataloader = state.data.calib

        self.initialize_obcq(modifiable_model)
        self.apply_obcq(calibration_dataloader)

        return True

    def initialize_obcq(self, model: "ModifiableModel"):
        """
        Setup for SparseGPT, initialize the the compressible layers of model, and set
        the device

        :param model: PyTorch model to sparsify
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.layer_prefix_ = model.layer_prefix
        self.model = self.model.model
        self.layer_compressors = []
        self._infer_mask_block_size()

        for idx, (name, layer) in enumerate(self.compressible_layers_.items()):
            _LOGGER.info(f"Preparing {name} for SparseGPT compression")
            layer_sparsity = (
                self.sparsity[idx] if isinstance(self.sparsity, List) else self.sparsity
            )
            args = {
                "sparsity": layer_sparsity,
                "prunen": self.prunen_,
                "prunem": self.prunem_,
                "blocksize": self.block_size,
                "percdamp": self.dampening_frac,
                "sequential_update": self.sequential_update,
                "quantize": self.quantize,
            }
            compressor = LayerCompressor(self.model, layer, idx, name, args)
            compressor.pre_compress()
            self.layer_compressors.append(compressor)

    @torch.no_grad()
    def apply_obcq(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run OBCQ on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for OBCQ
        """

        _LOGGER.info(f"Running SparseGPT calibration with {len(dataloader)} samples...")
        run_calibration_forward(self.model, dataloader)

        num_layers = len(self.compressible_layers_)
        for idx, layer_compressor in enumerate(self.layer_compressors):
            layer_sparsity = layer_compressor.args["sparsity"]
            _LOGGER.info(
                f"\n===== Compressing layer {idx+1}/{num_layers} "
                f"to sparsity {layer_sparsity} ====="
            )

            # Prune/quantize using SparseGPT
            layer_compressor.compress()
            layer_compressor.post_compress()

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param state: un-used, for matching spec of Modifier base class
        """
        #@torch.no_grad()
        #def apply_weights(module):
        #    if hasattr(module, "pruned_weight"):
        #        module.weight -= module.weight
        #        module.weight += module.pruned_weight
        #self.model.apply(apply_weights)

        for layer_compressor in self.layer_compressors:
            _LOGGER.info(f"Cleaning up {layer_compressor.name}")
            layer_compressor.revert_layer_wrappers()

        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return True

    def _infer_mask_block_size(self):
        """
        Infer the mask block size from the mask structure.
        Parses mask_structure of the form N:M where N, M are integers that
        define a custom block shape; and sets prunen_ and prunem_ accordingly.

        :post-condition: prunen_ and prunem_ are set
        """
        if self.mask_structure is None:
            raise ValueError("mask_structure must be defined")

        self.prunen_, self.prunem_ = list(map(int, self.mask_structure.split(":")))
