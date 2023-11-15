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


from typing import List, Union

from sparseml.core import Modifier
from sparseml.core.state import State
from sparseml.utils import ALL_TOKEN


__all__ = ["WandaPruningModifier"]


class WandaPruningModifier(Modifier):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695
    """

    sparsity: Union[float, List[float]]
    targets: Union[str, List[str], None] = ALL_TOKEN
    mask_structure: str = "unstructured"

    def on_initialize_structure(self, state: State, **kwargs):
        pass  # nothing needed for this modifier

    def compressible_layers(self) -> List:
        """
        Retrieves the modules corresponding to a list of
        compressible layer names

        :return: list of Pytorch modules to compress
        """
        compressible_dict = self.model.get_layers(self.targets)
        return [v for _, v in compressible_dict.items()]
