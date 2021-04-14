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
Functionality related to describing availability and information of sparsification
algorithms to models within in the ML frameworks.

The file is executable and will get the sparsification info for a given framework:

##########
Command help:
usage: info.py [-h] [--path PATH] framework

Compile the available setup and information for the sparsification of a model
in a given framework.

positional arguments:
  framework    the ML framework or path to a framework file to load the
               sparsification info for

optional arguments:
  -h, --help   show this help message and exit
  --path PATH  A full file path to save the sparsification info to. If not
               supplied, will print out the sparsification info to the
               console.

#########
EXAMPLES
#########

##########
Example command for getting the sparsification info for pytorch.
python src/sparseml/sparsification/info.py pytorch
"""

import argparse
import logging
import os
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from sparseml.base import execute_in_sparseml_framework
from sparseml.utils import clean_path, create_parent_dirs


__all__ = [
    "ModifierType",
    "ModifierPropInfo",
    "ModifierInfo",
    "SparsificationInfo",
    "sparsification_info",
    "save_sparsification_info",
    "load_sparsification_info",
]


_LOGGER = logging.getLogger(__name__)


class ModifierType(Enum):
    """
    Types of modifiers for grouping what functionality a Modifier falls under.
    """

    general = "general"
    training = "training"
    pruning = "pruning"
    quantization = "quantization"
    act_sparsity = "act_sparsity"
    misc = "misc"


class ModifierPropInfo(BaseModel):
    """
    Class for storing information and associated metadata for a
    property on a given Modifier.
    Extends pydantics BaseModel class for serialization to and from json
    in addition to proper type checking on construction.
    """

    name: str = Field(
        title="name",
        description=(
            "Name of the property for a Modifier. "
            "It can be accessed by this name on the modifier instance."
        ),
    )
    description: str = Field(
        title="description",
        description="Description and information for the property for a Modifier.",
    )
    type_: str = Field(
        title="type_",
        description=(
            "The format type for the property for a Modifier such as "
            "int, float, str, etc."
        ),
    )
    restrictions: Optional[List[Any]] = Field(
        default=None,
        title="restrictions",
        description=(
            "Value restrictions for the property for a Modifier. "
            "If set, restrict the set value to one of the contained restrictions."
        ),
    )


class ModifierInfo(BaseModel):
    """
    Class for storing information and associated metadata for a given Modifier.
    Extends pydantics BaseModel class for serialization to and from json
    in addition to proper type checking on construction.
    """

    name: str = Field(
        title="name",
        description=(
            "Name/class of the Modifier to be used for construction and identification."
        ),
    )
    description: str = Field(
        title="description",
        description="Description and info for the Modifier and what its used for.",
    )
    type_: ModifierType = Field(
        default=ModifierType.misc,
        title="type_",
        description=(
            "The type the given Modifier is for grouping by similar functionality."
        ),
    )
    props: List[ModifierPropInfo] = Field(
        default=[],
        title="props",
        description="The properties for the Modifier that can be set and controlled.",
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        title="warnings",
        description=(
            "Any warnings that apply for the Modifier and using it within a system"
        ),
    )


class SparsificationInfo(BaseModel):
    """
    Class for storing the information for sparsifying in a given framework.
    Extends pydantics BaseModel class for serialization to and from json
    in addition to proper type checking on construction.
    """

    modifiers: List[ModifierInfo] = Field(
        default=[],
        title="modifiers",
        description="A list of the information for the available modifiers",
    )

    def type_modifiers(self, type_: ModifierType) -> List[ModifierInfo]:
        """
        Get the contained Modifiers for a specific ModifierType.
        :param type_: The ModifierType to filter the returned list of Modifiers by.
        :type type_: ModifierType
        :return: The filtered list of Modifiers that match the given type_.
        :rtype: List[ModifierInfo]
        """
        modifiers = []

        for mod in self.modifiers:
            if mod.type_ == type_:
                modifiers.append(mod)

        return modifiers


def sparsification_info(framework: Any) -> SparsificationInfo:
    """
    Get the available setup for sparsifying model in the given framework.

    :param framework: The item to detect the ML framework for.
        See :func:`detect_framework` for more information.
    :type framework: Any
    :return: The sparsification info for the given framework
    :rtype: SparsificationInfo
    """
    _LOGGER.debug("getting sparsification info for framework %s", framework)
    info: SparsificationInfo = execute_in_sparseml_framework(
        framework, "sparsification_info"
    )
    _LOGGER.info("retrieved sparsification info for framework %s: %s", framework, info)

    return info


def save_sparsification_info(framework: Any, path: Optional[str] = None):
    """
    Save the sparsification info for a given framework.
    If path is provided, will save to a json file at that path.
    If path is not provided, will print out the info.

    :param framework: The item to detect the ML framework for.
        See :func:`detect_framework` for more information.
    :type framework: Any
    :param path: The path, if any, to save the info to in json format.
        If not provided will print out the info.
    :type path: Optional[str]
    """
    _LOGGER.debug(
        "saving sparsification info for framework %s to %s",
        framework,
        path if path else "sys.out",
    )
    info = (
        sparsification_info(framework)
        if not isinstance(framework, SparsificationInfo)
        else framework
    )

    if path:
        path = clean_path(path)
        create_parent_dirs(path)

        with open(path, "w") as file:
            file.write(info.json())

        _LOGGER.info(
            "saved sparsification info for framework %s in file at %s", framework, path
        ),
    else:
        print(info.json(indent=4))
        _LOGGER.info("printed out sparsification info for framework %s", framework)


def load_sparsification_info(load: str) -> SparsificationInfo:
    """
    Load the sparsification info from a file or raw json.
    If load exists as a path, will read from the file and use that.
    Otherwise will try to parse the input as a raw json str.

    :param load: Either a file path to a json file or a raw json string.
    :type load: str
    :return: The loaded sparsification info.
    :rtype: SparsificationInfo
    """
    load_path = clean_path(load)

    if os.path.exists(load_path):
        with open(load_path, "r") as file:
            load = file.read()

    info = SparsificationInfo.parse_raw(load)

    return info


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compile the available setup and information for the sparsification "
            "of a model in a given framework."
        )
    )
    parser.add_argument(
        "framework",
        type=str,
        help=(
            "the ML framework or path to a framework file to load the "
            "sparsification info for"
        ),
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "A full file path to save the sparsification info to. "
            "If not supplied, will print out the sparsification info to the console."
        ),
    )

    return parser.parse_args()


def _main():
    args = _parse_args()
    save_sparsification_info(args.framework, args.path)


if __name__ == "__main__":
    _main()
