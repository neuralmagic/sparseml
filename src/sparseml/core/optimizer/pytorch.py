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

from typing import Union, List, Any, Dict

from sparseml.core.optimizer.base import ModifiableOptimizer

from torch.optim import Optimizer

__all__ = ["ModifiableOptimizerPyTorch"]


class ModifiableOptimizerPyTorch(ModifiableOptimizer[Optimizer, Dict[str, Any]]):
    def get_param_groups(self) -> List[Dict[str, Any]]:
        return self.optimizer.param_groups

    def set_param_groups(self, param_groups: List[Dict[str, Any]]):
        self.optimizer.param_groups = param_groups

    def get_learning_rate(
        self, group_idx: Union[int, None] = None
    ) -> Union[float, List[float]]:
        if group_idx is not None:
            return self.optimizer.param_groups[group_idx]["lr"]
        return [group["lr"] for group in self.optimizer.param_groups]

    def set_learning_rate(self, lr: float, group_idx: Union[int, None] = None):
        if group_idx is not None:
            self.optimizer.param_groups[group_idx]["lr"] = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def get_attribute(
        self, attr_name: str, group_idx: Union[int, None] = None
    ) -> Union[Any, List[Any]]:
        if group_idx is not None:
            return self.optimizer.param_groups[group_idx].get(attr_name, None)
        return [group.get(attr_name, None) for group in self.optimizer.param_groups]

    def set_attribute(
        self, attr_name: str, value: Any, group_idx: Union[int, None] = None
    ):
        if group_idx is not None:
            self.optimizer.param_groups[group_idx][attr_name] = value
        else:
            for param_group in self.optimizer.param_groups:
                param_group[attr_name] = value
