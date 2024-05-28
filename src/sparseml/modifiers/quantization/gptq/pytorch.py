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
from sparseml.modifiers.quantization.gptq.base import GPTQModifier
from sparseml.modifiers.quantization.gptq.utils.gptq_wrapper import GPTQWrapper
from sparseml.modifiers.utils.layer_compressor import LayerCompressor
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward
from sparseml.utils.fsdp.context import fix_fsdp_module_name


__all__ = ["GPTQModifierPyTorch"]

_LOGGER = logging.getLogger(__name__)


class GPTQModifierPyTorch(GPTQModifier):
    """
    Pytorch implementation of GPTQ
    Lifecycle:
        - on_initialize
            - initialize_compression()
                - compressible_layers()
                - LayerCompressor.pre_compress()
            - apply_compression()
                - run_calibration_forward()
                - LayerCompressor.compress()
                - LayerCompressor.post_compress()
                - LayerCompressor.revert_layer_wrappers()
    | Sample yaml:
    | test_stage:
    |    obcq_modifiers:
    |      GPTQModifier:
    |          sequential_update: True
    |          dampening_frac: 0.001
    |          block_size: 128
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 8
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: "tensor"
    |                    group_size: 128


    :param model: Pytorch model to perform GPTQ on, in place.
    """

    model: Optional[ModifiableModel] = None
    layer_compressors_: Optional[List[Any]] = None

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)
        if self.quantization_modifier_:
            self.quantization_modifier_.initialize(state, **kwargs)
        if not self.quantize:
            raise ValueError("To use the GPTQModifier, quantization must be enabled.")

        modifiable_model = state.model
        calibration_dataloader = state.data.calib

        if self.targets is None:
            # if no targets are provided, default to the modules that shouldn't be
            # split by FSDP. For Transformers models this is equivalent to the
            # decoder layers (ie LlamaDecoderLayer)
            self.targets = modifiable_model.get_no_split_params()

        self.initialize_compression(modifiable_model, calibration_dataloader)
        self.apply_compression(calibration_dataloader)

        return True

    def initialize_compression(
        self,
        model: ModifiableModel,
        dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
    ):
        """
        Setup for GPTQ, initializes the model
        and other parameters, also initilializes the
        compressible layers of model, and sets the device

        :param model: model to initialize for compression
        :param dataloader: calibration data for GPTQ
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.model = self.model.model
        self.layer_compressors_ = []

        for idx, (name, layer) in enumerate(self.compressible_layers_.items()):
            name = fix_fsdp_module_name(name)
            _LOGGER.info(f"Preparing {name} for compression")
            args = self._pruning_arguments()
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
        Run GPTQ on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for GPTQ
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
            _LOGGER.info(f"\n===== Compressing layer {idx+1}/{num_layers} " " =====")

            # Prune/quantize using GPTQ
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

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return super(GPTQModifierPyTorch, self).on_finalize(state, **kwargs)

    def _pruning_arguments(self):
        """
        Gather the parameters needed for root module compression in a dict

        :return: dict of params for pruning
        """
        return {
            "blocksize": self.block_size,
            "percdamp": self.dampening_frac,
        }

    def _compression_class(self):
        """
        :return: wrapper class used for root modules of this compression class
        """
        return GPTQWrapper
