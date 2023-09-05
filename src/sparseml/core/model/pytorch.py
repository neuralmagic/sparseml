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

from typing import Dict, List, Tuple, Union

from torch.nn import Module, Parameter

from sparseml.core.model.base import ModifiableModel
from sparseml.utils.pytorch import (
    get_layer,
    get_layers,
    get_param,
    get_params,
    set_layer,
    set_param,
)


__all__ = ["ModifiableModelPyTorch"]


class ModifiableModelPyTorch(ModifiableModel[Module, Module, Parameter]):
    def get_layers(self, targets: Union[str, List[str]]) -> Dict[str, Module]:
        return get_layers(targets, self.model)

    def get_layer(self, target: str) -> Tuple[str, Module]:
        return get_layer(target, self.model)

    def set_layer(self, target: str, layer: Module) -> Module:
        return set_layer(target, layer, self.model)

    def get_params(self, targets: Union[str, List[str]]) -> Dict[str, Parameter]:
        return get_params(targets, self.model)

    def get_param(self, target: str) -> Tuple[str, Parameter]:
        return get_param(target, self.model)

    def set_param(self, target: str, param: Parameter):
        return set_param(target, param, self.model)
