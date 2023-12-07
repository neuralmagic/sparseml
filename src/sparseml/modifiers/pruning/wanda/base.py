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


from typing import List, Optional, Union

from sparseml.core import Modifier
from sparseml.core.model.base import ModifiableModel
from sparseml.core.state import State
from sparseml.utils import ALL_TOKEN


__all__ = ["WandaPruningModifier"]


class WandaPruningModifier(Modifier):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695

    Life-cycle:
        - initialze
            - compress
        - finalize

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    """

    sparsity: Union[float, List[float]]
    mask_structure: str = "0:0"
    targets: Union[str, List[str], None] = ALL_TOKEN
    compressible_layers_: Optional[List] = None

    def on_initialize_structure(self, state: State, **kwargs):
        """
        This modifier does not alter the model structure.
        This method is a no-op.

        :param state: Unused, kept to conform to the parent method signature
        :param kwargs: Unused, kept to conform to the parent method signature
        """

    def compressible_layers(self) -> List:
        """
        Retrieves the modules corresponding to a list of
        compressible layer names

        :precondition: self.model is set and is a `ModifiableModel`
        :precondition: The `ModifiableModel` implements a `get_layers`
            method
        :return: list of modules to compress
        """
        if not isinstance(self.model, ModifiableModel):
            raise ValueError(
                "`self.model` must be a ModifiableModel to use "
                f"the WANDA modifier but got {type(self.model)} instead"
            )

        compressible_dict = self.model.get_layers(self.targets)
        return [v for _, v in compressible_dict.items()]

    def _validate_layerwise_sparsity(self):
        if isinstance(self.sparsity, float):
            # single sparsity will be applied to all layers
            return

        if not isinstance(self.targets, List):
            raise ValueError(
                "Layer targets must be a list when specifying layer-wise"
                f" sparsity. Got {type(self.targets)}"
            )

        if len(self.targets) != len(self.sparsity):
            raise ValueError(
                "Number of layer targets must match the number of "
                f"sparsities. Got {len(self.targets)} layers and "
                f"{len(self.sparsity)} sparsities"
            )

        for layer_name in self.targets:
            if layer_name.startswith("re:"):
                raise ValueError(
                    "Using regular expressions for layer-wise sparsity "
                    f"profiles is not permitted. Found {layer_name}"
                )