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
from typing import List, Optional

import torch
from torch.quantization import QuantWrapper


__all__ = ["get_leaf_operations", "is_quantized", "get_quantization_scheme"]


def get_leaf_operations(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Get the leaf operations in the model
    (those that do not have operations as children)

    :param model: the model to get the leaf operations from
    :return: a list of the leaf operations
    """
    leaf_operations = []
    children = list(model.children())

    if children == []:
        return model
    else:
        for child in children:
            if isinstance(child, QuantWrapper):
                # if QuantWrapper encountered, treat the wrapped
                # module as a leaf operation
                leaf_operations.append(child.module)
                continue
            try:
                leaf_operations.extend(get_leaf_operations(child))
            except TypeError:
                leaf_operations.append(get_leaf_operations(child))
    return leaf_operations


def is_quantized(operation: torch.nn.Module) -> bool:
    """
    Check whether the operation is quantized (contains
    a quantization scheme)
    """
    return hasattr(operation, "quantization_scheme")


def get_quantization_scheme(
    operation: torch.nn.Module,
) -> Optional["QuantizationScheme"]:  # noqa F821
    """
    Get the quantization scheme of the operation.
    If the operation is not quantized, return None.

    :param operation: the operation to get the quantization scheme from
    :return: the quantization scheme of the operation or None if not quantized
    """
    if hasattr(operation, "quantization_scheme"):
        return operation.quantization_scheme
    return None
