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

from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from torch.nn import Module, Parameter

from sparseml.core.framework import Framework
from sparseml.core.model.base import ModelParameterizedLayer, ModifiableModel
from sparseml.pytorch.utils.sparsification_info.module_sparsification_info import (
    ModuleSparsificationInfo,
)
from sparseml.utils.pytorch import (
    get_layer,
    get_layers,
    get_layers_params,
    get_matching_layer,
    get_no_split_params,
    get_param,
    get_params,
    qat_active,
    set_layer,
    set_param,
)


__all__ = ["ModifiableModelPyTorch"]


class ModifiableModelPyTorch(ModifiableModel[Module, Module, Parameter]):
    """
    A ModifiableModel implementation for PyTorch models

    :param framework: the framework the model is in
    :param model: the model object
    :param layer_prefix: name of model attribute that contains the list of layers, i.e.
        model.decoder for OPT or just model for Llama
    """

    def __init__(
        self,
        framework: Optional[Framework] = None,
        model: Optional[Module] = None,
        layer_prefix: Optional[str] = None,
    ):
        super().__init__(framework=framework, model=model, layer_prefix=layer_prefix)

    def get_layers_params(
        self, targets: Union[str, List[str]]
    ) -> Dict[str, ModelParameterizedLayer[Module, Parameter]]:
        """
        :param targets: the target layers to get the parameters for
        :return: a dictionary of layer name to ModelParameterizedLayer
            instances for the given targets
        """
        return get_layers_params(targets, self.model)

    def get_layers(self, targets: Union[str, List[str]]) -> Dict[str, Module]:
        """
        :returns: a dictionary of layer name to layer for the given targets
        """
        return get_layers(targets, self.model)

    def get_layer(self, target: str) -> Tuple[str, Module]:
        """
        :returns: the layer for the given target
        """
        return get_layer(target, self.model)

    def set_layer(self, target: str, layer: Module) -> Module:
        """
        :param target: the target to set the layer for
        """
        return set_layer(target, layer, self.model)

    def get_params(self, targets: Union[str, List[str]]) -> Dict[str, Parameter]:
        """
        :param targets: the target parameters to get, can be a single target or
            a list of targets
        :return: a dictionary of parameter name to parameter for the given targets
        """
        return get_params(targets, self.model)

    def get_param(self, target: str) -> Tuple[str, Parameter]:
        """
        :returns: a tuple of the parameter name and its Parameter instance
            for the given target
        """
        return get_param(target, self.model)

    def set_param(self, target: str, param: Parameter):
        """
        :param target: the target to set the parameter for
        :param param: the parameter to set
        """
        return set_param(target, param, self.model)

    def loggable_items(self) -> Generator[Tuple[str, Any], None, None]:
        """
        PyTorch specific logging info for the model.
        loggable items are defined in the `ModuleSparsificationInfo` class,
        and include sparsity, quantization, and pruning information.

        This includes:
            - Total operation counts
            - Total parameter counts
            - sparsity percentages per layer (with non-zero sparsity only)
            - quantization bitwidth (for quantized layers)

        :return a generator that yields a tuple of:
            - the name of the loggable item
            - the value of the loggable item
        """
        sparsification_info = ModuleSparsificationInfo.from_module(self.model)

        yield from sparsification_info.loggable_items(
            percentages_only=True,
            non_zero_only=True,
            enabled_only=True,
        )

    def get_matching_layer(
        self, target: str, name_to_match: str, model: Module
    ) -> Optional[Tuple[str, Module]]:
        """
        :param target: regex layer name to target when searching model
        :param name_to_match: name to match targets to
        :param model: model to search for targets
        """
        return get_matching_layer(target, name_to_match, model)

    def qat_active(self) -> bool:
        """
        Checks if quantization aware training is set up in the model

        :return: True if QAT is active in any layer, False otherwise
        """
        return qat_active(self.model)

    def get_no_split_params(self) -> Union[str, List[str]]:
        """
        Get list of module classes that shouldn't be split when sharding. For
        Hugging Face Transformer models, this is the decoder layer type. For other
        types of models, this just returns all module names.

        :return: list of class names that shouldn't be split
        """
        return get_no_split_params(self.model)
