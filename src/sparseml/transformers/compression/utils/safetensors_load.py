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

import json
import os
import re
import struct
from typing import Dict, List

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


__all__ = [
    "get_safetensors_header",
    "match_param_name",
    "merge_names",
    "get_weight_mappings",
    "get_nested_weight_mappings",
]


def get_safetensors_header(safetensors_path: str) -> Dict[str, str]:
    """
    Extracts the metadata from a safetensors file as JSON

    :param safetensors_path: path to a safetensors file
    :return: dictionary of metadata extracted from the safetensors file
    """
    with open(safetensors_path, "rb") as f:
        length_of_header = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(length_of_header)
        header = json.loads(header_data)

    return header


def match_param_name(full_name: str, param_name: str) -> str:
    """
    Helper function extracting the uncompressed parameterized layer name from a
    compressed name. Assumes the compressed name was merged using merge_names.

    :param full_name: full name of parameter in compressed model
    :param param_name: compression paramater name
    :return: uncompressed name of the uncompressed parameterized layer
    """
    pattern = r"^(.*)\." + param_name + r"$"
    regex = re.findall(pattern, full_name)
    if len(regex) == 0:
        return None
    return regex[0]


def merge_names(parent_name: str, child_name: str) -> str:
    """
    Helper function for merging an uncompressed parameterized layer name with a
    compression parameter. Names merged with this function can then be parsed by
    match_param_name.

    :param parent_name: uncompressed parameterized layer name
    :param child_name: compression parameter name
    :return: merged compressed name
    """
    return parent_name + "." + child_name


def get_weight_mappings(model_path: str) -> Dict[str, str]:
    """
    Takes a path to a state dict saved in safetensors format and returns a mapping
    from parameterized layer name to file location.

    {
        layer.weight.bitmask: file_location,
        layer.weight.row_offsets: file_location,
        layer.weight.shape: file_location,
        layer.weight.compressed: file_location
    }

    This generalizes to cases where the model is split into multiple safetensors files

    :param model_path: path to safetensors state dict, must contain either a single
    safetensors file or multiple files with an index
    :return: mapping of parameterized layer name to file location
    """
    safetensors_path = os.path.join(model_path, SAFE_WEIGHTS_NAME)
    index_path = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(safetensors_path):
        # we have a single safetensors file to read
        header = get_safetensors_header(safetensors_path)
        for key in header.keys():
            header[key] = SAFE_WEIGHTS_NAME
        header.pop("__metadata__", None)
    elif os.path.exists(index_path):
        # we have multiple safetensors file, read from index
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        header = index["weight_map"]

    # convert weight locations to full paths
    for key, value in header.items():
        header[key] = os.path.join(model_path, value)

    return header


def get_nested_weight_mappings(
    model_path: str, params_to_nest: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Takes a path to a state dict saved in safetensors format and returns a nested
    mapping from uncompressed parameterized layer names to the file locations of each
    of the layers compression parameters.

    layer.weight: {
        bitmask: file_location,
        row_offsets: file_location,
        shape: file_location,
        compressed: file_location
    }

    This generalizes to cases where the model is split into multiple safetensors files

    :param model_path: path to safetensors state dict, must contain either a single
    safetensors file or multiple files with an index
    :return: nested mapping of parameterized layer name to file location
    """
    weight_mappings = get_weight_mappings(model_path)

    nested_weight_mappings = {}
    for key in weight_mappings.keys():
        for param_name in params_to_nest:
            maybe_match = match_param_name(key, param_name)
            if maybe_match is not None:
                dense_param = maybe_match
                if dense_param not in nested_weight_mappings:
                    nested_weight_mappings[dense_param] = {}
                nested_weight_mappings[dense_param][param_name] = weight_mappings[key]

    return nested_weight_mappings
