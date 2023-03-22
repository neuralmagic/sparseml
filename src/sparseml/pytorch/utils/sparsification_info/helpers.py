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
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field


__all__ = ["get_leaf_operations", "is_quantized", "get_dtype", "unpack_dictionary"]


def unpack_dictionary(
    dictionary: Dict[str, Any], tag="", separator="/"
) -> Generator[Tuple[str, Any], None, None]:
    """
    Given a dictionary, unpack it into a generator of tag, item pairs,
    where tag is the string name of the item.

    :param dictionary: the dictionary to unpack
    :param tag: the tag to prepend to the item name
    :param separator: the separator to separate tags on varying nesting depths
    """
    for name, item in dictionary.items():
        if isinstance(item, BaseModel):
            item = dict(item)
        if isinstance(item, dict):
            yield from unpack_dictionary(item, tag=f"{tag}{separator}{name}")
        else:
            yield f"{tag}{separator}{name}", item


def get_leaf_operations(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Get the leaf operations in the model
    (those that do not have operations as children)

    :param model: the model to get the leaf operations from
    :return: a list of the leaf operations
    """
    return [
        submodule
        for submodule in model.modules()
        if len(list(submodule.children())) == 0
    ]


def is_quantized(operation: torch.nn.Module) -> bool:
    """
    Check whether the operation is quantized
    """
    if hasattr(operation, "qscheme"):
        return operation.qscheme is not None
    return False


class QuantizationDtype(BaseModel):
    """
    Model for the quantization dtype of a model
    """

    activation: Optional[torch.dtype] = Field(
        default=None, description="The quantization dtype of activation function"
    )
    weight: Optional[torch.dtype] = Field(
        default=None, description="The quantization dtype of weight function"
    )

    class Config:
        arbitrary_types_allowed = True


def get_dtype(operation: torch.nn.Module) -> QuantizationDtype:
    """
    Get the quantization dtype of the operation (both activation and weight).
    If the operation is not quantized, return None for activation dtype and
    original dtype of weight.

    :param operation: the operation to get the quantization dtype from
    :return: the quantization dtype of the operation
    """
    if hasattr(operation, "qconfig"):
        if operation.qconfig is None:
            pass
        else:
            return QuantizationDtype(
                activation=operation.qconfig.activation.p.keywords["dtype"],
                weight=operation.qconfig.weight.p.keywords["dtype"],
            )
    elif hasattr(operation, "weight"):
        return QuantizationDtype(weight=operation.weight.dtype)

    return QuantizationDtype()
