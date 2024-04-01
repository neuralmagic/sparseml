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

from sparseml.transformers.sparsification.modification.base import (
    check_transformers_version,
)
from sparsezoo.utils.registry import RegistryMixin


class ModificationRegistry(RegistryMixin):
    """
    A registry for modification functions that can be applied to models
    so that they can be used in the context of sparseml.transformers
    """

    @classmethod
    def get_value_from_registry(cls, name: str):
        """
        Extends the base class method to check the transformers version after
        successfully retrieving the value from the registry. The motivation is
        to ensure that the transformers version falls within the supported range
        before we proceed with model modification.
        """
        retrieved_value = super().get_value_from_registry(name)
        check_transformers_version()
        return retrieved_value
