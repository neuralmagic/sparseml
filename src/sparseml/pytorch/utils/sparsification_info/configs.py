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

from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field


class SparsificationSummaries(BaseModel):
    quantization: Tuple[int, float] = Field(
        description="A tuple that displays of the number of "
        "layers/the percent of layers that are quantized."
    )
    pruning: Tuple[int, float] = Field(
        description="A tuple that displays of the number of "
        "layers/the percent of layers that are pruned."
    )

    @classmethod
    def from_dict(cls, module_information: Dict[str, Any]):
        quantization, pruning, ops = SparsificationSummaries.compute_summaries(
            module_information
        )
        return cls(
            quantization=quantization,
            pruning=pruning,
            ops=ops,
        )

    @staticmethod
    def compute_summaries(
        module_information: Dict[str, Any]
    ) -> Tuple[Tuple[int, float], Tuple[int, float], Dict[str, int]]:
        """ """
        count_quantized_layers = 0
        count_pruned_layers = 0
        count_all_layers = 0
        counts_ops = {}

        for layer_name, layer_param in module_information.items():
            count_all_layers += 1
            if layer_param["is_quantized"]:
                count_quantized_layers += 1
            if layer_param["is_sparse"]:
                count_pruned_layers += 1

        return (
            (count_quantized_layers, count_quantized_layers / count_all_layers),
            (count_pruned_layers, count_pruned_layers / count_all_layers),
            counts_ops,
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
    def from_dict(cls, module_information: Dict[str, Any]):
        zero_count, zero_count_percent = SparsificationPruning.zero_weight_count(
            module_information
        )
        return cls(
            zero_count=zero_count,
            zero_count_percent=zero_count_percent,
        )

    @staticmethod
    def zero_weight_count(module_information: Dict[str, Any]) -> Tuple[int, float]:
        """ """
        zero_count = {
            layer_name: layer_info["num_zero_elements"]
            for layer_name, layer_info in module_information.items()
        }
        zero_count_percent = {
            layer_name: layer_info["num_zero_elements"] / layer_info["num_elements"]
            for layer_name, layer_info in module_information.items()
        }
        return zero_count, zero_count_percent


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
    def from_dict(
        cls, module_information: Dict[str, Any]
    ) -> "SparsificationQuantization":
        enabled = {
            layer_name: layer_info["is_quantized"]
            for layer_name, layer_info in module_information.items()
        }
        dtype = {
            layer_name: layer_info["dtype"]
            for layer_name, layer_info in module_information.items()
        }

        return cls(
            enabled=enabled,
            dtype=dtype,
        )


class SparsificationDistillation(BaseModel):
    losses: Dict[str, str] = Field(
        description="A dictionary that maps the name of a layer "
        "to the loss function used for that layer."
    )
