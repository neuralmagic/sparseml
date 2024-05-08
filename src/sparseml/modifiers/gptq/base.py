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

from sparseml.core import Modifier
from sparseml.core.factory import ModifierFactory
from sparseml.core.model.base import ModifiableModel
from sparseml.core.state import State


__all__ = ["GPTQModifier"]

_LOGGER = logging.getLogger(__name__)


class GPTQModifier(Modifier):
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
                - LayerCompressor.revert_layer_wrappers()
        - on_finalize

    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during GPTQ, or '__ALL__'
        to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Whether or not to quantize weights. Set to
        True to quantize using an existing quantization modifier, or pass in the
        configuration for a quantization modifier if one does not already exist
        in the recipe
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    """

    sequential_update: Optional[bool] = False
    targets: Union[str, List[str], None] = None
    compressible_layers_: Optional[List] = None
    block_size: int = 128
    quantize: Union[bool, Dict] = True
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
                    "GPTQ quantization is set to False, but a "
                    "quantization modifier is already active on the model "
                    "resetting quantize to True"
                )
                self.quantize = True
            elif self.quantize and not quantization_already_active:
                _LOGGER.warning(
                    "GPTQ quantization is set to True without an "
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
                    "GPTQModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"type {type(self.quantize)}"
                )
            if len(self.quantize) != 1:
                raise ValueError(
                    "GPTQModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"{len(self.quantize)} modifiers"
                )
            if quantization_already_active:
                _LOGGER.warning(
                    "Attempting to initialize quantization for GPTQ "
                    "but a quantization modifier has already been applied. "
                    "The quantization configuration defined under the "
                    "GPTQ modifier will be ignored."
                )
                self.quantize = True
                return
            self._build_quant_modifier_from_dict(self.quantize, state.framework)
            self.quantize = True

        if self.quantization_modifier_:
            self.quantization_modifier_.on_initialize_structure(state, **kwargs)

    def compressible_layers(self) -> Dict:
        """
        Retrieves the modules corresponding to a list of
        compressible layer names

        :precondition: self.model is set and is a `ModifiableModel`
        :precondition: The `ModifiableModel` implements a `get_layers`
            method
        :return: dictionary of modules to compress
        """
        if not isinstance(self.model, ModifiableModel):
            raise ValueError(
                "`self.model` must be a ModifiableModel to use "
                f"the {self.__class__.__qualname__} modifier but got "
                f"{type(self.model)} instead"
            )

        return self.model.get_layers(self.targets)

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

    def on_finalize(self, state: State, **kwargs):
        """
        Nothing to do on finalize, on this level.
        Quantization Modifier if any will be finalized in the subclass

        :param state: session state storing input model and calibration data
        :param kwargs: additional arguments
        :return: True
        """
        return True
