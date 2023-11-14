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
from sparseml.modifiers.obcq.utils.helpers import cache_attention_inputs
from sparseml.modifiers.obcq.utils.layer_compressor import LayerCompressor


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
    device_: str = "cuda:0"
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
        device = state.hardware.device

        self.initialize_obcq(modifiable_model, device)
        self.apply_obcq(calibration_dataloader)

        return True

    def initialize_obcq(
        self,
        model: "ModifiableModel",
        device: Optional[str] = "cuda:0",
    ):
        """
        Setup for SparseGPT, initialize the the compressible layers of model, and set
        the device

        :param model: PyTorch model to sparsify
        :param device: device to run sparsification on, preferably a GPU
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.layer_prefix_ = model.layer_prefix
        self.model = self.model.model
        self._set_device(device)
        self._infer_mask_block_size()

    @torch.no_grad()
    def apply_obcq(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run OBCQ on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for OBCQ
        """
        accum_kwargs = {"dataloader": dataloader}

        # Step 0: Pass the calibration data through the (compressed) bottom part of the
        # network, capturing the outputs which will become the inputs to the first
        # decoder layer. Also return attention_mask as part of kwargs
        extras = self.compress_bottom(
            dev=self.device_,
            target_ids=self.target_ids,
            layer_prefix=self.layer_prefix_,
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
            layer_sparsity = (
                self.sparsity[idx] if isinstance(self.sparsity, List) else self.sparsity
            )
            _LOGGER.info(
                f"\n===== Compressing layer {idx+1}/{num_layers} "
                f"to sparsity {layer_sparsity} ====="
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
            layer_compressor = LayerCompressor(self.model, layer, idx, inputs, args)

            # Prune/quantize using SparseGPT
            layer_kwargs = layer_compressor.compress(dev=self.device_, **accum_kwargs)
            accum_kwargs.update(layer_kwargs)

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param state: un-used, for matching spec of Modifier base class
        """

        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return True

    def compress_bottom(
        self,
        dataloader: List = None,
        nsamples: int = None,
        dev: str = "cuda:0",
        target_ids: List[str] = None,
        layer_prefix: Optional[str] = None,
    ) -> Dict:
        """
        Runs calibration data through the bottom part of the network (everything up
        to the first decoder layer) and return the captured outputs

        :param dataloader: calibration data to pass through the model
        :param nsamples: number of samples to use for calibration, or None to use it all
        :param dev: device to use
        :param target_ids: list of keys in model output to cache, NOTE: this argument
            has been deprecated and will be removed in a future release
        :param layer_prefix: name of model attribute that contains the list of layers,
            i.e. model.decoder for OPT or just model for Llama
        :return: outputs from bottom part of network, attention mask, and kv-cache state
        """
        layer_prefix = layer_prefix or self.layer_prefix_
        cached_inputs = cache_attention_inputs(
            model=self.model,
            dataloader=dataloader,
            device=dev,
            nsamples=nsamples,
            target_ids=target_ids,
            layer_prefix=layer_prefix,
        )

        outputs = cached_inputs.pop("inputs")
        outputs = [o[0] for o in outputs]
        cached_inputs.update({"outputs": outputs})
        return cached_inputs

    def _set_device(self, device: str):
        if "cuda" in device and not torch.cuda.is_available():
            self.device_ = "cpu"
        else:
            self.device_ = device

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
