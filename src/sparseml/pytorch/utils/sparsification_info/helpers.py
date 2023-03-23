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
from pydantic import BaseModel
from torch.quantization import QuantWrapper


__all__ = [
    "get_leaf_operations",
    "is_quantized",
    "get_quantization_scheme",
    "unpack_dictionary",
    "reformat",
]


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


def reformat(tag: str) -> Optional[str]:
    """
    Enforce a consistent format for the tags to be logged

    :param tag: the tag to reformat
    :return: the reformatted tag
    """
    tag_components = tag.split("/")[1:]

    if tag_components[0] == "summary_info":
        tag_components = _reformat_summary_info(tag_components)

    elif tag_components[0] == "pruning_info":
        tag_components = _reformat_pruning_info(tag_components)

    elif tag_components[0] == "quantization_info":
        tag_components = _reformat_quantization_info(tag_components)

    return "/".join(tag_components) if tag_components else None


def _rename_component(components, old_name, new_name):
    return [new_name if name == old_name else name for name in components]


def _capitalize_last_component(components):
    return components[:-1] + [components[-1].capitalize()]


def _swap_components(components, first_name, second_name):
    first_index = components.index(first_name)
    second_index = components.index(second_name)
    components[first_index], components[second_index] = (
        components[second_index],
        components[first_index],
    )
    return components


def _reformat_summary_info(tag_components: List[str]) -> List[str]:
    tag_components = _rename_component(
        tag_components, old_name="summary_info", new_name="SparsificationSummaries"
    )
    if "quantized" in tag_components:
        tag_components = _rename_component(
            tag_components, old_name="quantized", new_name="_Quantization"
        )
        tag_components = _capitalize_last_component(tag_components)
        return tag_components

    elif "pruned" in tag_components:
        tag_components = _rename_component(
            tag_components, old_name="pruned", new_name="_Sparsity"
        )
        tag_components = _capitalize_last_component(tag_components)
        return tag_components

    elif "parameter_counts" in tag_components:
        tag_components = _rename_component(
            tag_components, old_name="parameter_counts", new_name="Params/Counts"
        )
        return tag_components

    elif "operation_counts" in tag_components:
        tag_components = _rename_component(
            tag_components, old_name="operation_counts", new_name="Ops/Counts"
        )
        return tag_components


def _reformat_pruning_info(tag_components: List[str]) -> List[str]:
    tag_components = _rename_component(
        tag_components, old_name="pruning_info", new_name="PruningSummaries"
    )
    tag_components = _capitalize_last_component(tag_components)
    tag_components = _swap_components(
        tag_components, "zero_parameters", tag_components[-1]
    )

    return tag_components[:-1]


def _reformat_quantization_info(tag_components: List[str]) -> List[str]:
    tag_components = _rename_component(
        tag_components,
        old_name="quantization_info",
        new_name="QuantizationSummaries",
    )
    if "quantization_schema" in tag_components:
        if "num_bits" not in tag_components:
            return False
        else:
            # e.g ['QuantizationSummaries', 'quantization_schema',
            #       'ConvBnReLU2d', 'input_activations', 'num_bits']
            # to  ['QuantizationSummaries', 'Input_activations',
            #      'Num_bits', 'ConvBnReLU2d']
            tag_components = [tag_components[i] for i in [0, 3, 4, 2]]
            tag_components[1] = tag_components[1].capitalize()
            tag_components[2] = tag_components[2].capitalize()

    return tag_components
