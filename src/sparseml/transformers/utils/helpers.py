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
Helper variables and functions for integrating SparseML with huggingface/transformers
flows
"""
import logging
from typing import Any, Dict, List


__all__ = [
    "RECIPE_NAME",
    "extract_metadata_from_args",
]


RECIPE_NAME = "recipe.yaml"
_LOGGER = logging.getLogger(__name__)


def extract_metadata_from_args(metadata_args: List[str], args: Dict[str, Any]) -> Dict:
    metadata = {}
    for arg in metadata_args:
        if arg not in args.keys():
            _LOGGER.warning(
                f"Required metadata argument {arg} was not found "
                f"in the training arguments. Setting {arg} to None."
            )
            metadata[arg] = None
        else:
            metadata[arg] = args[arg]

    return metadata
