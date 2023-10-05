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
from typing import Dict, Generic, List, TypeVar, Union

from sparseml.core.framework_object import MultiFrameworkObject


__all__ = ["ModifiableModel", "ModelParameterizedLayer"]


MT = TypeVar("MT")
LT = TypeVar("LT")
PT = TypeVar("PT")


@dataclass
class ModelParameterizedLayer(Generic[LT, PT]):
    layer_name: str
    layer: LT
    param_name: str
    param: PT


@dataclass
class ModifiableModel(Generic[MT, LT, PT], MultiFrameworkObject):
    model: MT = None

    def __init__(self, framework=None, model=None):
        self.model = model

    def get_layers_params(
        self, targets: Union[str, List[str]]
    ) -> Dict[str, ModelParameterizedLayer[LT, PT]]:
        raise NotImplementedError()

    def get_layers(self, targets: Union[str, List[str]]) -> Dict[str, LT]:
        raise NotImplementedError()

    def get_layer(self, target: str) -> LT:
        raise NotImplementedError()

    def set_layer(self, target: str, layer: LT):
        raise NotImplementedError()

    def get_params(self, targets: Union[str, List[str]]) -> Dict[str, PT]:
        raise NotImplementedError()

    def get_param(self, target: str) -> PT:
        raise NotImplementedError()

    def set_param(self, target: str, param: PT):
        raise NotImplementedError()
