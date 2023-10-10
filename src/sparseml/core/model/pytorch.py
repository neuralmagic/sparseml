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

from typing import Dict, List, Optional, Tuple, Union

from torch.nn import Module, Parameter

from sparseml.core.framework import Framework
from sparseml.core.model.base import ModelParameterizedLayer, ModifiableModel
from sparseml.utils.pytorch import (
    get_layer,
    get_layers,
    get_layers_params,
    get_param,
    get_params,
    set_layer,
    set_param,
)


__all__ = ["ModifiableModelPyTorch"]


class ModifiableModelPyTorch(ModifiableModel[Module, Module, Parameter]):
    """
    A ModifiableModel implementation for PyTorch models

    :param framework: the framework the model is in
    :param model: the model object
    """

    def __init__(
        self, framework: Optional[Framework] = None, model: Optional[Module] = None
    ):
        super().__init__(framework=framework, model=model)

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
