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

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch.nn
from pydantic import BaseModel, Field


def is_quantized(operation: torch.nn.Module) -> bool:
    if hasattr(operation, "qconfig"):
        return operation.qconfig is not None
    return False


def get_dtype(operation: torch.nn.Module) -> Any:
    if hasattr(operation, "weight"):
        return operation.weight.dtype
    return None


class SparsificationSummaries(BaseModel):
    quantization: Tuple[int, float] = Field(
        description="A tuple that displays of the number of "
        "layers/the percent of layers that are quantized."
    )
    pruning: Tuple[int, float] = Field(
        description="A tuple that displays of the number of "
        "layers/the percent of layers that are pruned."
    )
    ops: Dict[str, int] = Field(
        description="A dictionary that maps the name of an operation "
        "to the number of times that operation is used in the model."
    )

    @classmethod
    def from_module_info(
        cls,
        param_information: Dict[str, Any],
        operations: List[torch.nn.Module],
        is_pruned_threshold: Tuple[float, float] = (0.1, 0.99),
    ) -> "SparsificationSummaries":

        # Compute quantization summary statistics
        num_quantized_layers = len([op for op in operations if is_quantized(op)])
        percentage_quantized_layers = num_quantized_layers / len(operations)

        # Compute pruning summary statistics
        lower_treshold = min(is_pruned_threshold)
        upper_treshold = max(is_pruned_threshold)
        num_pruned_layers = 0
        for param_info in param_information.values():
            if (
                lower_treshold
                <= param_info["percentage_zero_weights"]
                <= upper_treshold
            ):
                num_pruned_layers += 1
        percentage_pruned_layers = num_pruned_layers / len(param_information.keys())

        # Compute ops summary statistics
        operations = [op.__class__.__name__ for op in operations]

        return cls(
            quantization=(num_quantized_layers, percentage_quantized_layers),
            pruning=(num_pruned_layers, percentage_pruned_layers),
            ops=Counter(operations),
        )


class SparsificationPruning(BaseModel):
    zero_count_percent: Dict[str, float] = Field(
        description="A dictionary that maps the name of a layer "
        "to the percent of weights that are zeroed out "
        "in that layer."
    )
    zero_count: Dict[str, int] = Field(
        description="A dictionary that maps the name of a layer "
        "to the number of weights that are zeroed out "
        "in that layer."
    )

    @classmethod
    def from_module_info(cls, param_information: Dict[str, Any]):
        zero_count = {
            layer_name: layer_info["num_zero_elements"]
            for layer_name, layer_info in param_information.items()
        }
        zero_count_percent = {
            layer_name: layer_info["percentage_zero_weights"]
            for layer_name, layer_info in param_information.items()
        }
        return cls(
            zero_count=zero_count,
            zero_count_percent=zero_count_percent,
        )


class SparsificationQuantization(BaseModel):
    enabled: Dict[str, bool] = Field(
        description="A dictionary that maps the name of a layer "
        "to whether or not that layer is quantized."
    )
    dtype: Dict[str, Any] = Field(
        description="A dictionary that maps the name of a layer"
        "to the dtype (precision) of that layer."
    )

    @classmethod
    def from_module_info(
        cls, operations: List[torch.nn.Module]
    ) -> "SparsificationQuantization":

        enabled = {op.__class__.__name__: is_quantized(op) for op in operations}
        dtype = {op.__class__.__name__: get_dtype(op) for op in operations}

        return cls(enabled=enabled, dtype=dtype)


class SparsificationDistillation(BaseModel):
    losses: Optional[Any] = None
