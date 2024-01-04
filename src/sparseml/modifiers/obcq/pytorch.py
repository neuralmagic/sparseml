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

    Lifecycle: TODO update
        - on_initialize
            - setup
                - compressible_layers
            - prune
                - compress_bottom
                - LayerCompressor.compress
        - on_finalize

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

        return super(SparseGPTModifierPyTorch, self).on_initialize(state, **kwargs)

    def _pruning_arguments(self, sparsity):
        return {
            "sparsity": sparsity,
            "prunen": self.prunen_,
            "prunem": self.prunem_,
            "blocksize": self.block_size,
            "percdamp": self.dampening_frac,
        }

    def _compression_class(self):
        return SparseGptWrapper

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the observers used by the OBCQ algorithm and set kv-cache configuration

        :param state: un-used, for matching spec of Modifier base class
        """
        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return super(SparseGPTModifierPyTorch, self).on_finalize(state, **kwargs)
