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

from typing import Dict, Optional

from torch import Tensor
from torch.nn import Module

import sparseml
from compressed_tensors import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.quantization.utils import is_model_quantized
from sparseml.pytorch.utils import ModuleSparsificationInfo


class SparsityConfigMetadata:
    """
    Class of helper functions for filling out a SparsityCompressionConfig with readable
    metadata from the model
    """

    @staticmethod
    def infer_global_sparsity(
        model: Module, state_dict: Optional[Dict[str, Tensor]] = None
    ) -> float:
        """
        Calculates the global percentage of sparse zero weights in the model

        :param model: pytorch model to infer sparsity of
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        :return: global sparsity of model
        """

        info = ModuleSparsificationInfo(model, state_dict=state_dict)
        global_sparsity = info.params_sparse_percent
        return global_sparsity

    @staticmethod
    def infer_sparsity_structure() -> str:
        """
        Determines what sparsity structure, if any, was applied in the currently active
        sparse session

        :return: sparsity structure as a string
        """
        current_session = sparseml.active_session()
        stage_modifiers = current_session.lifecycle.modifiers
        sparsity_structure = "unstructured"

        # check for applied pruning modifiers
        for stage in stage_modifiers:
            if stage.applied:
                for modifier in stage.modifiers:
                    if hasattr(modifier, "mask_structure"):
                        sparsity_structure = modifier.mask_structure
                        break

        return sparsity_structure

    @staticmethod
    def from_pretrained(
        model: Module,
        state_dict: Optional[Dict[str, Tensor]] = None,
        compress: bool = False,
    ) -> Optional["SparsityCompressionConfig"]:
        """
        Determines compression type and informational parameters for a given model

        :param model: pytorch model to calculate sparsity config for
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        :param compress: whether or not to compress the model on disk
        :return: compression config inferred from the model
        """

        global_sparsity = SparsityConfigMetadata.infer_global_sparsity(
            model, state_dict=state_dict
        )

        if global_sparsity < 0.05:
            return None

        sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure()
        if is_model_quantized(model):
            # compressing a sparse quantized model is not supported yet
            format = CompressionFormat.dense.value
        elif compress:
            format = CompressionFormat.sparse_bitmask.value
        else:
            format = CompressionFormat.dense.value

        return SparsityCompressionConfig.load_from_registry(
            format,
            global_sparsity=global_sparsity,
            sparsity_structure=sparsity_structure,
        )

    @staticmethod
    def fill_config_details(
        config: SparsityCompressionConfig,
        model: Module,
        state_dict: Optional[Dict[str, Tensor]] = None,
    ):
        """
        Fills in informational sparsity parameters from a given model

        :param config: sparsity config to fill in
        :param model: pytorch model to infer config parameters from
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        """
        config.global_sparsity = SparsityConfigMetadata.infer_global_sparsity(
            model, state_dict=state_dict
        )
        config.sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure()
