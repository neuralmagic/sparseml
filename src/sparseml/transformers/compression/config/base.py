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

from typing import Optional

from pydantic import BaseModel
from torch.nn import Module

import sparseml.core.session as session_manager
from sparseml.pytorch.utils import ModuleSparsificationInfo
from sparsezoo.utils.registry import RegistryMixin


__all__ = ["CompressionConfig"]


class CompressionConfig(RegistryMixin, BaseModel):
    """
    Base data class for storing compression parameters

    :param format: name of compression format
    """

    format: str

    @staticmethod
    def infer_global_sparsity(model: Module) -> float:
        info = ModuleSparsificationInfo(model)
        global_sparsity = info.params_sparse_percent
        return global_sparsity

    @staticmethod
    def infer_sparsity_structure() -> str:
        current_session = session_manager.active_session()
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
    def infer_config_from_model(
        model: Module, force_dense: bool = False
    ) -> Optional["CompressionConfig"]:

        global_sparsity = CompressionConfig.infer_global_sparsity(model)

        if global_sparsity < 0.05:
            return None

        sparsity_structure = CompressionConfig.infer_sparsity_structure()
        if force_dense:
            format = "dense_sparsity"
        else:
            format = "sparse_bitmask"

        return CompressionConfig.load_from_registry(
            format,
            global_sparsity=global_sparsity,
            sparsity_structure=sparsity_structure,
        )

    def fill_config_details(self, model: Module):
        self.global_sparsity = CompressionConfig.infer_global_sparsity(model)
        self.sparsity_structure = CompressionConfig.infer_sparsity_structure()
