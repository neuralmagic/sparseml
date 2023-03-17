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

from typing import Iterable, Tuple

import torch
from pydantic import BaseModel

from sparseml.pytorch.utils.sparsification_info.configs import (
    SparsificationDistillation,
    SparsificationPruning,
    SparsificationQuantization,
    SparsificationSummaries,
)


class ModuleSparsificationInfo(BaseModel):
    summary_info: SparsificationSummaries
    pruning_info: SparsificationPruning
    quantization_info: SparsificationQuantization
    distillation_info: SparsificationDistillation

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> "ModuleSparsificationInfo":
        raise NotImplementedError()

    def loggable_items(self) -> Iterable[Tuple[str, float]]:
        raise NotImplementedError()
