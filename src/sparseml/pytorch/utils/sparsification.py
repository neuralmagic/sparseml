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

"""
Helper functions for retrieving information related to model sparsification
"""

import json
from typing import Dict

import torch
from torch.nn import Module

from sparseml.pytorch.utils.helpers import (
    get_prunable_layers,
    get_quantizable_layers,
    get_quantized_layers,
    tensor_sparsity,
)


__all__ = ["ModuleSparsificationInfo"]


class ModuleSparsificationInfo:
    def __init__(self, module: Module):
        self.module = module
        self.trainable_params = list(
            filter(lambda param: param.requires_grad, self.module.parameters())
        )

    def __str__(self):
        return json.dumps({})

    @property
    def params_total(self) -> int:
        return sum(torch.numel(param) for param in self.trainable_params)

    @property
    def params_sparse(self) -> int:
        return sum(tensor_sparsity(param) for param in self.trainable_params)

    @property
    def params_sparse_percent(self) -> float:
        return self.params_sparse / float(self.params_total) * 100

    @property
    def params_prunable_total(self) -> int:
        return sum(
            torch.numel(layer.weight)
            for (name, layer) in get_prunable_layers(self.module)
        )

    @property
    def params_pruanble_sparse(self) -> int:
        return sum(
            tensor_sparsity(layer.weight)
            for (name, layer) in get_prunable_layers(self.module)
        )

    @property
    def params_prunable_sparse_percent(self) -> float:
        return self.params_pruanble_sparse / float(self.params_prunable_total) * 100

    @property
    def params_quantizable(self) -> int:
        return sum(
            torch.numel(layer.weight)
            + (torch.numel(layer.bias) if hasattr(layer, "bias") and layer.bias else 0)
            for (name, layer) in get_quantizable_layers(self.module)
        )

    @property
    def params_quantized(self) -> int:
        return sum(
            torch.numel(layer.weight)
            + (torch.numel(layer.bias) if hasattr(layer, "bias") and layer.bias else 0)
            for (name, layer) in get_quantized_layers(self.module)
        )

    @property
    def params_quantized_percent(self) -> float:
        return self.params_quantized / float(self.params_quantizable) * 100

    @property
    def params_info(self) -> Dict[str, Dict]:
        return {
            f"{name}.weight": {
                "numel": torch.numel(layer.weight),
                "sparsity": tensor_sparsity(layer.weight).item(),
                "quantized": hasattr(layer, "weight_fake_quant"),
            }
            for (name, layer) in get_prunable_layers(self.module)
        }
