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

from sparseml.core.model.base import ModifiableModel
from sparseml.core.state import State
from sparseml.modifiers.pruning.wanda.base import WandaPruningModifier
from sparseml.modifiers.pruning.wanda.utils.wanda_wrapper import WandaWrapper
from sparseml.modifiers.utils.layer_compressor import LayerCompressor
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward


_LOGGER = logging.getLogger(__name__)


class WandaPruningModifierPyTorch(WandaPruningModifier):
    """
    Pytorch implementation of WandaPruningModifier

    Lifecycle:
        - on_initialize
            - initialize_compression()
                - compressible_layers()
                - LayerCompressor.pre_compress()
            - apply_compression()
                - run_calibration_forward()
                - LayerCompressor.compress()
                - LayerCompressor.post_compress()
        - on_finalize
            - LayerCompressor.revert_layer_wrappers()

    :param model: `ModifiableModel` to perform WANDA on, in-place
    """

    model: Optional[ModifiableModel] = None
    layer_compressors_: List = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the WANDA algorithm on the current state

        :param state: session state storing input model and calibration data
        :param kwargs: Unused, kept to conform to the parent method signature
        """
        self._validate_layerwise_sparsity()

        modifiable_model = state.model
        calibration_dataloader = state.data.calib

        self.initialize_compression(modifiable_model)
        self.apply_compression(calibration_dataloader)

        return True

    def initialize_compression(self, model: ModifiableModel):
        """
        Setup for WANDA, initializes the model, device,
        and other parameters, also initilializes the
        compressible layers of model, and sets the device

        :param model: model to initialize for compression
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.model = self.model.model
        self.layer_compressors_ = []
        self._infer_mask_block_size()

        for idx, (name, layer) in enumerate(self.compressible_layers_.items()):
            _LOGGER.info(f"Preparing {name} for compression")
            layer_sparsity = (
                self.sparsity[idx] if isinstance(self.sparsity, List) else self.sparsity
            )
            args = self._pruning_arguments(layer_sparsity)
            comp_cls = self._compression_class()
            compressor = LayerCompressor(comp_cls, self.model, layer, idx, name, args)
            if not self.sequential_update:
                # add all batch processing hooks before the forward pass
                compressor.pre_compress()
            self.layer_compressors_.append(compressor)

    @torch.no_grad()
    def apply_compression(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run Wanda on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for WANDA
        """
        class_name = self.__class__.__name__.replace("PyTorch", "")
        _LOGGER.info(
            f"Running {class_name} calibration with " f"{len(dataloader)} samples..."
        )
        if not self.sequential_update:
            # in non-sequential mode we run one forward batch for all modules
            run_calibration_forward(self.model, dataloader, mask_padding=True)

        num_layers = len(self.compressible_layers_)
        for idx, layer_compressor in enumerate(self.layer_compressors_):
            layer_sparsity = layer_compressor.args["sparsity"]
            _LOGGER.info(
                f"\n===== Compressing layer {idx+1}/{num_layers} "
                f"to sparsity {layer_sparsity} ====="
            )

            # Prune/quantize using SparseGPT
            if self.sequential_update:
                # in sequential mode we run one forward pass for each module we
                # want to compress, this will be really slow but allows compression in
                # earlier layers to affect later layers
                layer_compressor.pre_compress()
                _LOGGER.info(f"Calibrating {layer_compressor.name}...")
                run_calibration_forward(self.model, dataloader, mask_padding=True)
            layer_compressor.compress()
            layer_compressor.post_compress()
            layer_compressor.revert_layer_wrappers()
            torch.cuda.empty_cache()

    def on_finalize(self, state: State, **kwargs):
        """
        Nothing to clean up for this module

        :param state: Unused, kept to conform to the parent method signature
        :param kwargs: Unused, kept to conform to the parent method signature
        """

        return True

    def _pruning_arguments(self, sparsity) -> Dict[str, Any]:
        """
        Gather the parameters needed for root module compression in a dict

        :param sparsity: target sparsity
        :return: dict of params for pruning
        """
        return {
            "sparsity": sparsity,
            "prunen": self.prunen_,
            "prunem": self.prunem_,
        }

    def _compression_class(self):
        """
        :return: wrapper class used for root modules of this compression class
        """
        return WandaWrapper

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
