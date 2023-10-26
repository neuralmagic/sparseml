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
from typing import Any, Generic, List, Optional, TypeVar, Union

from sparseml.core.framework_object import MultiFrameworkObject


__all__ = ["ModifiableOptimizer"]


OT = TypeVar("OT")
PGT = TypeVar("PGT")


@dataclass
class ModifiableOptimizer(Generic[OT, PGT], MultiFrameworkObject):
    """
    Generic class to wrap an optimizer

    :param optimizer: the optimizer to wrap
    :param attach_optim_callbacks: if True, will attach callbacks to the optimizer
    :param framework: the framework the optimizer is for
    """

    optimizer: Optional[OT] = None

    def __init__(self, optimizer=None, attach_optim_callbacks=False, framework=None):
        self.optimizer = optimizer

    def get_param_groups(self) -> List[PGT]:
        """
        :return: the parameter groups for the optimizer
        """
        raise NotImplementedError()

    def set_param_groups(self, param_groups: List[PGT]):
        """
        Set the parameter groups for the optimizer

        :param param_groups: the parameter groups to set
        """
        raise NotImplementedError()

    def get_learning_rate(
        self, group_index: Union[int, None] = None
    ) -> Union[float, List[float]]:
        """
        Get learning rate for the given group index or all groups
        from the optimizer

        :param group_index: the index of the group to get the learning rate for,
            set to None to get all learning rates
        :return: the learning rate or list of learning rates
        """
        raise NotImplementedError()

    def set_learning_rate(self, lr: float, group_index: Union[int, None] = None):
        """
        Set the learning rate for the given group index or all groups
        """
        raise NotImplementedError()

    def get_attribute(
        self, name: str, group_index: Union[int, None] = None
    ) -> Union[Any, List[Any]]:
        """
        Get the attribute for the given group index or all groups
        from the optimizer

        :param name: the name of the attribute to get
        :param group_index: the index of the group to get the attribute for,
            set to None to get all attributes
        :return: the attribute value or the list of attribute values
        """
        raise NotImplementedError()

    def set_attribute(
        self, name: str, value: Any, group_index: Union[int, None] = None
    ):
        """
        Set the attribute for the given group index or all groups in the optimizer

        :param name: the name of the attribute to set
        :param value: the value to set the attribute to
        :param group_index: the index of the group to set the attribute for,
            set to None to set attribute in all groups
        """
        raise NotImplementedError()
