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

from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from pydantic import BaseModel
from torch.nn.parameter import Parameter

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
    distillation_info: Optional[SparsificationDistillation]

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> "ModuleSparsificationInfo":
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "module must be a torch.nn.Module, not {}".format(type(module))
            )
        module_information = defaultdict()
        # iterate over parameters of the module
        for name, param in module.named_parameters():
            # param is a weight or bias
            # op is conv/linear named_modules
            module_information[name] = ModuleSparsificationInfo.get_param_info(param)

        return cls(
            summary_info=SparsificationSummaries.from_dict(module_information),
            pruning_info=SparsificationPruning.from_dict(module_information),
            quantization_info=SparsificationQuantization.from_dict(module_information),
            distillation_info=None,
        )

    def loggable_items(self) -> Iterable[Tuple[str, float]]:
        raise NotImplementedError()

    @staticmethod
    def get_param_info(param: Parameter) -> Dict[str, Any]:
        return {
            "num_elements": param.numel(),
            "num_zero_elements": param.numel() - param.count_nonzero().item(),
            "is_sparse": param.is_sparse,
            "is_quantized": param.is_quantized,
            "dtype": param.dtype,
        }
