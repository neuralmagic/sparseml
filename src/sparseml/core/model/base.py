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

from dataclasses import dataclass
from typing import Any, Dict, Generator, Generic, List, Optional, Tuple, TypeVar, Union

from sparseml.core.framework import Framework
from sparseml.core.framework_object import MultiFrameworkObject


__all__ = ["ModifiableModel", "ModelParameterizedLayer"]


MT = TypeVar("MT")
LT = TypeVar("LT")
PT = TypeVar("PT")


@dataclass
class ModelParameterizedLayer(Generic[LT, PT]):
    """
    A dataclass for holding a parameter and its layer

    :param layer_name: the name of the layer
    :param layer: the layer object
    :param param_name: the name of the parameter
    :param param: the parameter object
    """

    layer_name: str
    layer: LT
    param_name: str
    param: PT


@dataclass
class ModifiableModel(Generic[MT, LT, PT], MultiFrameworkObject):
    """
    A MultiFrameWorkObject for holding a model. Also defines the
    contract that must be followed for framework specific implementations.

    Automatically instantiates the correct subclass object based on the
    specified framework if it exists. If the framework is not specified,
    the default "general" framework will be used. The inheritors of this class
    must be named in the following format: ModifiableModel{framework.class_name()}
    to be searchable by the MultiFrameworkObject factory method.

    :param framework: the framework the model is in
    :param layer_prefix: name of model attribute that contains the list of layers, i.e.
        model.decoder for OPT or just model for Llama
    :param model: the model object
    """

    model: MT = None

    def __init__(
        self,
        framework: Optional[Framework] = None,
        model=None,
        layer_prefix: Optional[str] = None,
    ):
        self.model = model
        self._layer_prefix = layer_prefix

    def get_layers_params(
        self, targets: Union[str, List[str]]
    ) -> Dict[str, ModelParameterizedLayer[LT, PT]]:
        """
        :param targets: the targets to get the layers and params for
        :return: a dictionary of the layer name to ModelParameterizedLayer instance
            for that layer
        """
        raise NotImplementedError()

    def get_layers(self, targets: Union[str, List[str]]) -> Dict[str, LT]:
        """
        :param targets: the targets to get the layers for
        :return: a dictionary of the layer name to layer instance for that layer
        """
        raise NotImplementedError()

    def get_layer(self, target: str) -> LT:
        """
        :param target: the target to get the layer for
        :return: the layer instance corresponding to the target
        """
        raise NotImplementedError()

    def set_layer(self, target: str, layer: LT):
        """
        :param target: the target to set the layer for
        :param layer: the layer instance to set
        """
        raise NotImplementedError()

    def get_params(self, targets: Union[str, List[str]]) -> Dict[str, PT]:
        """
        :param targets: the targets to get the params for
        :return: a dictionary of the param name to param instance for that param
        """
        raise NotImplementedError()

    def get_param(self, target: str) -> PT:
        """
        :param target: the target to get the param for
        :return: the param instance corresponding to the target
        """
        raise NotImplementedError()

    def set_param(self, target: str, param: PT):
        """
        :param target: the target to set the param for
        :param param: the param instance to set
        """
        raise NotImplementedError()

    def loggable_items(self) -> Generator[Tuple[str, Any], None, None]:
        """
        Model level information to be logged for the model

        :return a generator that yields a tuple of:
            - the name of the loggable item
            - the value of the loggable item
        """

    @property
    def layer_prefix(self) -> Optional[str]:
        """
        :return: the name of model attribute that contains the list of layers, i.e.
            model.decoder for OPT or just model for Llama
        """
        return self._layer_prefix

    @layer_prefix.setter
    def layer_prefix(self, value: Optional[str]):
        """
        :param value: the name of model attribute that contains the list of layers, i.e.
            model.decoder for OPT or just model for Llama
        """
        self._layer_prefix = value

    def get_matching_layer(
        self, target: str, name_to_match: str, model: LT
    ) -> Optional[Tuple[str, LT]]:
        """
        :param target: regex layer name to target when searching model
        :param name_to_match: name to match targets to
        :param model: model to search for targets
        """
        raise NotImplementedError()

    def qat_active(self) -> bool:
        """
        Checks if quantization aware training is set up in the model

        :return: True if QAT is active in any layer, False otherwise
        """
        raise NotImplementedError()

    def get_no_split_params(self) -> Union[str, List[str]]:
        """
        Get list of module classes that shouldn't be split when sharding

        :return: list of class names that shouldn't be split
        """
        raise NotImplementedError()
