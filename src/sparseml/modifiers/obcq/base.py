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
from typing import Any, Dict, List, Optional, Union

from sparseml.core.factory import ModifierFactory
from sparseml.core.state import State
from sparseml.modifiers.pruning.wanda.base import WandaPruningModifier


__all__ = ["SparseGPTModifier"]

_LOGGER = logging.getLogger(__name__)


class SparseGPTModifier(WandaPruningModifier):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model

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

    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Whether or not to quantize weights during SparseGPT. Set to
        True to quantize using an existing quantization modifier, or pass in the
        configuration for a quantization modifier if one does not already exist
        in the recipe
    :param sparsity: Sparsity to compress model to
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    """

    block_size: int = 128
    quantize: Union[bool, Dict] = False
    sparsity: Union[float, List[float]] = 0.0
    dampening_frac: Optional[float] = 0.01
    quantization_modifier_: Any = None

    def on_initialize_structure(self, state: State, **kwargs):
        """
        Check the model's quantization state matches that expected by this modifier,
        adding a default quantization scheme if needed

        :param state: session state storing input model and calibration data
        """
        quantization_already_active = state.model.qat_active()
        if isinstance(self.quantize, bool):
            if not self.quantize and quantization_already_active:
                _LOGGER.warning(
                    "SparseGPT quantization is set to False, but a "
                    "quantization modifier is already active on the model "
                    "resetting quantize to True"
                )
                self.quantize = True
            elif self.quantize and not quantization_already_active:
                _LOGGER.warning(
                    "SparseGPT quantization is set to True without an "
                    "active quantization modifier. Creating a default "
                    "8-bit quantization modifier"
                )
                default_quant_config = {"QuantizationModifier": {}}
                self._build_quant_modifier_from_dict(
                    default_quant_config, state.framework
                )
            return  # use existing quantization modifier if there is one
        else:
            if not isinstance(self.quantize, Dict):
                raise ValueError(
                    "SparseGPTModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"type {type(self.quantize)}"
                )
            if len(self.quantize) != 1:
                raise ValueError(
                    "SparseGPTModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"{len(self.quantize)} modifiers"
                )
            if quantization_already_active:
                _LOGGER.warning(
                    "Attempting to initialize quantization for SparseGPT "
                    "but a quantization modifier has already been applied. "
                    "The quantization configuration defined under the "
                    "SparseGPT modifier will be ignored."
                )
                self.quantize = True
                return
            self._build_quant_modifier_from_dict(self.quantize, state.framework)
            self.quantize = True

        if self.quantization_modifier_:
            self.quantization_modifier_.on_initialize_structure(state, **kwargs)

    def _build_quant_modifier_from_dict(self, quant_config, framework):
        modifier_type = list(quant_config.keys())[0]
        modifier_args = quant_config[modifier_type]
        self.quantization_modifier_ = ModifierFactory.create(
            modifier_type,
            framework=framework,
            allow_registered=True,
            allow_experimental=True,
            **modifier_args,
        )
