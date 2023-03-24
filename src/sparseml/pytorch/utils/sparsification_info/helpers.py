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

from sparseml.pytorch.nn import Identity


__all__ = ["get_leaf_operations", "is_quantized", "get_quantization_scheme"]


def get_leaf_operations(
    model: torch.nn.Module,
    operations_to_skip: List[torch.nn.Module] = [Identity],
    operations_to_unwrap: List[torch.nn.Module] = [QuantWrapper],
) -> List[torch.nn.Module]:
    """
    Get the leaf operations in the model
    (those that do not have operations as children)

    :param model: the model to get the leaf operations from
    :param operations_to_skip: a list of leaf operations that will be
        omitted when getting the leaf operations
    :param operations_to_unwrap: a list of operations that will be unwrapped
        when getting the leaf operations. Unwrapping means that we directly
        add the module(s) that is/are wrapped by the operation to the list
        of leaf operations
    :return: a list of the leaf operations
    """
    leaf_operations = []
    children = list(model.children())

    if children == [] and model not in operations_to_skip:
        return model
    else:
        for child in children:
            if isinstance(child, tuple(operations_to_unwrap)):
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
