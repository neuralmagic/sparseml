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
from sparseml.core.state import State
from sparseml.utils import ALL_TOKEN


__all__ = ["SparseGPTModifier"]

_LOGGER = logging.getLogger(__name__)


class SparseGPTModifier(Modifier):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model

    Life-cycle:
        - initialze
            - compress
        - finalize

    :param sparsity: Sparsity to compress model to
    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Whether or not to quantize weights during SparseGPT. Set to True
    to quantize using an existing quantization modifier, or pass in the configuration
    for a quantization modifier if one does not already exist in the recipe
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param prunen: N for N:M pruning
    :param prunem: M for N:M pruning
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    :param target_ids: list of keys in model output to cache
    :param layer_prefix: name of model attribute that contains the list of layers, i.e.
        model.decoder for OPT or just model for Llama
    """

    sparsity: Union[float, List[float]]
    block_size: int
    quantize: Union[bool, Dict]
    dampening_frac: Optional[float] = 0.01
    sequential_update: Optional[bool] = True
    prunen: Optional[int] = 0
    prunem: Optional[int] = 0
    targets: Union[str, List[str], None] = ALL_TOKEN
    target_ids: Optional[List[str]] = None
    layer_prefix: Optional[str] = None
    compressible_layers_: List = None
    quantization_modifier_: Any = None

    def compressible_layers(self) -> List:
        """
        Retrieves the modules corresponding to a list of compressible layer names

        :return: list of Pytorch modules to compress
        """
        compressible_dict = self.model.get_layers(self.targets)
        return [v for _, v in compressible_dict.items()]

    def on_initialize_structure(self, state: State, **kwargs):
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

    def _validate_layerwise_sparisity(self):
        if isinstance(self.sparsity, float):
            return  # single sparsity will be applied to all layers

        if not isinstance(self.targets, List):
            raise ValueError(
                "Layer targets must be a list when specifying layer-wise"
                f" sparsity. Got {self.targets}"
            )

        if len(self.targets) != len(self.sparsity):
            raise ValueError(
                "Number of layer targets must match the number of "
                f"sparsities. Got {len(self.targets)} layers and "
                f"{len(self.sparsity)} sparsities"
            )

        for layer_name in self.targets:
            if layer_name.startswith("re:"):
                raise ValueError(
                    "Using regular expressions for layer-wise sparsity "
                    f"profiles is not permitted. Found {layer_name}"
                )
