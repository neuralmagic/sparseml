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
from sparseml.modifiers.obcq.utils.helpers import cache_attention_inputs
from sparseml.modifiers.pruning.wanda.base import WandaPruningModifier
from sparseml.modifiers.pruning.wanda.utils.layer_compressor import WandaLayerCompressor


_LOGGER = logging.getLogger(__name__)


class WandaPruningModifierPyTorch(WandaPruningModifier):
    """
    Pytorch implementation of WandaPruningModifier

    Lifecycle:
        - on_initialize
            - setup
                - compressible_layers
            - prune
                - compress_bottom
                - LayerCompressor.compress
        - on_finalize

    :param model: `ModifiableModel` to perform wanda on, in-place
    """

    model: Optional[ModifiableModel] = None
    device_: str = "cuda:0"
    layer_prefix_: Optional[str] = None
    prunen_: Optional[int] = None
    prunem_: Optional[int] = None
    layer_compressor_class_ = WandaLayerCompressor

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the WANDA algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        self._validate_layerwise_sparsity()
        self.setup(state=state, **kwargs)

        # run on calibration data
        self.prune(dataloader=state.data.calib)
        torch.cuda.empty_cache()
        return True

    def setup(self, state: State, **kwargs):
        """
        Setup for WANDA, initializes the model, device,
        and other parameters, also initilializes the
        compressible layers of model, and sets the device

        :param state: session state storing input model and calibration data
        """
        self.model = state.model
        self.compressible_layers_ = self.compressible_layers()
        self.device_ = self._set_device(device=state.hardware.device)
        self.layer_prefix_ = self.model.layer_prefix
        self._infer_mask_block_size()

    @torch.no_grad()
    def prune(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run Wanda on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for WANDA
        """
        accum_kwargs = {"dataloader": dataloader}
        pytorch_model = self.model.model

        # Step 0: Pass the calibration data through the (compressed) bottom part of the
        # network, capturing the outputs which will become the inputs to the first
        # decoder layer. Also return attention_mask as part of kwargs
        extras = self.compress_bottom(
            dev=self.device_,
            layer_prefix=self.layer_prefix_,
            **accum_kwargs,
        )
        accum_kwargs.update(extras)

        # Step 1: Sequentially prune decoder layers
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
            args = self._get_compression_args(layer_sparsity=layer_sparsity)
            # Prune using GPT
            layer_compressor = self.layer_compressor_class_(
                model=pytorch_model,
                layer=layer,
                layer_index=idx,
                inputs=inputs,
                args=args,
            )
            layer_kwargs = layer_compressor.compress(dev=self.device_, **accum_kwargs)
            accum_kwargs.update(layer_kwargs)

    def _get_compression_args(self, layer_sparsity):
        return {
            "sparsity": layer_sparsity,
            "prunen": self.prunen_,
            "prunem": self.prunem_,
        }

    def compress_bottom(
        self,
        dataloader: List = None,
        nsamples: int = None,
        dev: str = "cuda:0",
        layer_prefix: Optional[str] = None,
        target_ids: Optional[List[int]] = None,
    ) -> Dict:
        """
        Runs calibration data through the bottom part of the network (everything up
        to the first decoder layer) and return the captured outputs

        :param dataloader: calibration data to pass through the model
        :param nsamples: number of samples to use for calibration, or None to use it all
        :param dev: device to use
        :param layer_prefix: name of model attribute that contains the list of layers,
            i.e. model.decoder for OPT or just model for Llama
        :param target_ids: list of keys in model output to cache, NOTE: this argument
            has been deprecated and will be removed in a future release, also must be
            set to None for Wanda
        :return: outputs from bottom part of network, attention mask, and kv-cache state
        """
        layer_prefix = layer_prefix or self.layer_prefix_
        pytorch_model = self.model.model
        cached_inputs = cache_attention_inputs(
            model=pytorch_model,
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

    def on_finalize(self, state: State, **kwargs):
        return True

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
