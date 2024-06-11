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

from pydantic import Field

from compressed_tensors.quantization import (
    QuantizationScheme,
    is_preset_scheme,
    preset_name_to_scheme,
)
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
        - on_finalize
            - LayerCompressor.revert_layer_wrappers()


    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during GPTQ, or '__ALL__'
        to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Set to True to quantize using an existing quantization modifier,
        or pass in the configuration for a quantization modifier if one does not
        already exist in the recipe
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param config_groups: [Used, if a quantization modifier is not specified],
        dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param ignore: [Used, if a quantization modifier is not specified]
        optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param disable_quantization_observer_epoch: [Used, if a quantization modifier is
        not specified] Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    :param scheme: [Used, if a quantization modifier is not specified], the quantization
        scheme to apply to the model, this is a dictionary that supports all keys from
        QuantizationScheme except targets, which will be set to the targets parameter
        set at the modifier level. Can also be set to a dictionary of the format
        `preset_scheme_name: targets` for example: `W8A8: ['Linear']` for weight 8 bit
        or a string of a preset scheme if targets is provided
        and activation 8 bit quantization on the Linear layers.
    """

    sequential_update: Optional[bool] = False
    targets: Union[str, List[str], None] = None
    block_size: int = 128
    quantize: Union[bool, Dict] = True
    dampening_frac: Optional[float] = 0.01
    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    disable_quantization_observer_epoch: Optional[float] = None
    num_calibration_steps: Optional[int] = None
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    compressible_layers_: Optional[List] = None
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
                    "active quantization modifier."
                )
                self._build_quant_modifier(state.framework)
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

    def _build_quant_modifier(self, framework):
        """
        Build a quantization modifier based on the specified config_groups,
        ignore list, and num_calibration_steps.

        :postcondition: self.quantization_modifier_ is set to the built
            quantization modifier
        :param framework: the framework to build the quantization modifier for
        """

        quantization_args_names = [
            "config_groups",
            "num_calibration_steps",
            "ignore",
            "disable_quantization_observer_epoch",
        ]

        quant_args = {
            key: getattr(self, key)
            for key in quantization_args_names
            if getattr(self, key, False)
        }

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        if self.scheme is not None:
            # takes precedence over config_groups

            if isinstance(self.scheme, str) and is_preset_scheme(self.scheme):
                # attach targets to scheme
                self.scheme = {self.scheme: self.targets}

            quant_args["config_groups"] = {}
            for idx, key in enumerate(self.scheme.keys()):
                if is_preset_scheme(key):
                    scheme = preset_name_to_scheme(key, self.scheme[key])
                else:
                    scheme = QuantizationScheme.model_validate(
                        {"targets": self.scheme[key], **self.scheme}
                    )

                group_name = f"group_{idx}"
                quant_args["config_groups"][group_name] = scheme

        if "config_groups" not in quant_args or len("config_groups") == 0:
            default_quant_scheme = QuantizationScheme.default_scheme(
                targets=self.targets
            )
            quant_args["config_groups"] = {"group_0": default_quant_scheme}
        _LOGGER.info(f"Building quantization modifier with args: {quant_args}")
        vllm_quant_config = {"QuantizationModifier": quant_args}
        self._build_quant_modifier_from_dict(vllm_quant_config, framework)

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
