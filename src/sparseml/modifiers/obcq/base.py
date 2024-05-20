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

from typing import Dict, List, Optional, Union

from sparseml.core import Modifier
from sparseml.core.model.base import ModifiableModel
from sparseml.core.state import State


__all__ = ["SparseGPTModifier"]


class SparseGPTModifier(Modifier):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model

    Lifecycle:
        - on_initialize
            - initialize_compression()
                - compressible_layers()
                - LayerCompressor.pre_compress()
            - apply_compression()
                - run_calibration_forward()
                - LayerCompressor.compress()
                - LayerCompressor.post_compress()
        - on_finalize
            - LayerCompressor.revert_layer_wrappers()

    :param sparsity: Sparsity to compress model to
    :param sparsity_profile: Can be set to 'owl' to use Outlier Weighed
        Layerwise Sparsity (OWL), more information can be found
        in the paper https://arxiv.org/pdf/2310.05175
    :param owl_m: Number of outliers to use for OWL
    :param owl_lmbda: Lambda value to use for OWL
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param preserve_sparsity_mask: Whether or not to preserve the sparsity mask
        during when applying sparsegpt, this becomes useful when starting from a
        previously pruned model, defaults to False.
    """

    sparsity: Union[float, List[float]] = 0.0
    sparsity_profile: Optional[str] = None
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None
    mask_structure: str = "0:0"
    sequential_update: Optional[bool] = False
    targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    preserve_sparsity_mask: bool = False
    prunen_: Optional[int] = None
    prunem_: Optional[int] = None
    compressible_layers_: Optional[List] = None

    def on_initialize_structure(self, state: State, **kwargs):
        """
        Initialize the structure of the model for compression.
        This modifier does not modifiy the model structure, so this method
        is a no-op.

        :param state: session state storing input model and calibration data
        """
        return True

    def compressible_layers(self) -> Dict:
        """
        Retrieves the modules corresponding to a list of
        compressible layer names

        :precondition: self.model is set and is a `ModifiableModel`
        :precondition: The `ModifiableModel` implements a `get_layers`
            method
        :return: dictionary of modules to compress
        """
        if not isinstance(self.model, ModifiableModel):
            raise ValueError(
                "`self.model` must be a ModifiableModel to use "
                f"the {self.__class__.__qualname__} modifier but got "
                f"{type(self.model)} instead"
            )

        return self.model.get_layers(self.targets)

    def _validate_layerwise_sparsity(self):
        if isinstance(self.sparsity, float):
            # single sparsity will be applied to all layers
            return

        target_layers = list(self.compressible_layers_.keys())

        if len(target_layers) != len(self.sparsity):
            raise ValueError(
                "Number of layer targets must match the number of "
                f"sparsities. Got {len(target_layers)} layers and "
                f"{len(self.sparsity)} sparsities"
            )

    def on_finalize(self, state: State, **kwargs):
        """
        Nothing to do on finalize, on this level.
        Quantization Modifier if any will be finalized in the subclass

        :param state: session state storing input model and calibration data
        :param kwargs: additional arguments
        :return: True
        """
        return True
