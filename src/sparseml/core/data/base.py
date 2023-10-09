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
from typing import Generic, TypeVar

from sparseml.core.framework_object import MultiFrameworkObject


__all__ = ["ModifiableData"]

DT = TypeVar("DT")  # Dataset Type


@dataclass
class ModifiableData(Generic[DT], MultiFrameworkObject):
    """
    A base class for data that can be modified by modifiers.

    :param data: The data to be modified
    :param num_samples: The number of samples in the data
    """

    data: DT = None
    num_samples: int = None

    def get_num_batches(self) -> int:
        """
        :return: The number of batches in the data
        """
        raise NotImplementedError()

    def set_batch_size(self, batch_size: int):
        """
        :param batch_size: The new batch size to use
        """
        raise NotImplementedError()

    def get_batch_size(self) -> int:
        """
        :return: The current batch size
        """
        raise NotImplementedError()
