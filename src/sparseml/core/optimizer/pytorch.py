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

from typing import Any, Dict, List, Union

from torch.optim import Optimizer

from sparseml.core.optimizer.base import ModifiableOptimizer


__all__ = ["ModifiableOptimizerPyTorch"]


class ModifiableOptimizerPyTorch(ModifiableOptimizer[Optimizer, Dict[str, Any]]):
    """
    A ModifiableOptimizer implementation for PyTorch optimizers

    :param optimizer: the torch optimizer to wrap
    :param attach_optim_callbacks: if True, will attach callbacks to the optimizer
    :param framework: the framework the optimizer is for (PyTorch)
    """

    def __init__(self, optimizer=None, attach_optim_callbacks=False, framework=None):
        super().__init__(
            optimizer=optimizer,
            attach_optim_callbacks=attach_optim_callbacks,
            framework=framework,
        )

    def get_param_groups(self) -> List[Dict[str, Any]]:
        """
        Get the parameter groups for the optimizer

        :return: a list of dictionaries representing parameter groups for
            the optimizer
        """
        return self.optimizer.param_groups

    def set_param_groups(self, param_groups: List[Dict[str, Any]]):
        """
        Set the parameter groups for the optimizer

        :param param_groups: the parameter groups to set, must be a list of
            dictionaries representing parameter groups for the optimizer
        """
        self.optimizer.param_groups = param_groups

    def get_learning_rate(
        self, group_idx: Union[int, None] = None
    ) -> Union[float, List[float]]:
        """
        Get the learning rate for the given group index from the optimizer

        :param group_idx: the index of the group to get the learning rate for,
            set to None to get all learning rates
        :return: the learning rate or list of learning rates
        """
        if group_idx is not None:
            return self.optimizer.param_groups[group_idx]["lr"]
        return [group["lr"] for group in self.optimizer.param_groups]

    def set_learning_rate(self, lr: float, group_idx: Union[int, None] = None):
        """
        Set the learning rate for the given group index or all groups
        in the optimizer

        :param lr: the learning rate to set
        :param group_idx: the index of the group to set the learning rate for,
            set to None to set learning rates for all parameter groups
        """
        if group_idx is not None:
            self.optimizer.param_groups[group_idx]["lr"] = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def get_attribute(
        self, attr_name: str, group_idx: Union[int, None] = None
    ) -> Union[Any, List[Any]]:
        """
        Get the attribute value for the given group index from the optimizer

        :param attr_name: the name of the attribute to get
        :param group_idx: the index of the group to get the attribute for,
            set to None to get the attribute(s) value for all groups
        :return: the attribute value or list of attribute values
        """
        if group_idx is not None:
            return self.optimizer.param_groups[group_idx].get(attr_name, None)
        return [group.get(attr_name, None) for group in self.optimizer.param_groups]

    def set_attribute(
        self, attr_name: str, value: Any, group_idx: Union[int, None] = None
    ):
        """
        Set the attribute for the given group index or all groups in the optimizer

        :param attr_name: the name of the attribute to set
        :param value: the value to set the attribute to
        :param group_idx: the index of the group to set the attribute for,
            set to None to set attribute in all groups
        """
        if group_idx is not None:
            self.optimizer.param_groups[group_idx][attr_name] = value
        else:
            for param_group in self.optimizer.param_groups:
                param_group[attr_name] = value
