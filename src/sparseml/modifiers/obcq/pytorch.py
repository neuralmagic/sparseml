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

import logging
from typing import List, Optional

from sparseml.core.model import ModifiableModel
from sparseml.core.state import State
from sparseml.modifiers.obcq.base import SparseGPTModifier
from sparseml.modifiers.obcq.utils.sgpt_wrapper import SparseGptWrapper
from sparseml.modifiers.pruning.wanda.pytorch import WandaPruningModifierPyTorch


__all__ = ["SparseGPTModifierPyTorch"]

_LOGGER = logging.getLogger(__name__)


class SparseGPTModifierPyTorch(WandaPruningModifierPyTorch, SparseGPTModifier):
    """
    Pytorch implementation of SparseGPT

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

    :param model: Pytorch model to perform OBCQ on, in-place
    """

    model: Optional[ModifiableModel] = None
    layer_compressors: List = None

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)
        if self.quantization_modifier_:
            self.quantization_modifier_.initialize(state, **kwargs)
        if not self.quantize and self.sparsity == 0.0:
            raise ValueError(
                "To use the SparseGPTModifier, target sparsity must be > 0.0 or "
                "quantization must be enabled."
            )

        return super(SparseGPTModifierPyTorch, self).on_initialize(state, **kwargs)

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return super(SparseGPTModifierPyTorch, self).on_finalize(state, **kwargs)

    def _pruning_arguments(self, sparsity):
        """
        Gather the parameters needed for root module compression in a dict

        :param sparsity: target sparsity
        :return: dict of params for pruning
        """
        return {
            "sparsity": sparsity,
            "prunen": self.prunen_,
            "prunem": self.prunem_,
            "blocksize": self.block_size,
            "percdamp": self.dampening_frac,
        }

    def _compression_class(self):
        """
        :return: wrapper class used for root modules of this compression class
        """
        return SparseGptWrapper
