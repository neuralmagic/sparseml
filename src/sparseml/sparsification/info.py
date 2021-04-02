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
"""

import logging
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from sparseml.base import detect_framework, execute_in_sparseml_framework


__all__ = [
    "ModifierType",
    "ModifierPropInfo",
    "ModifierInfo",
    "SparsificationInfo",
    "sparsification_info",
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
    Load the available setup for sparsifying model in the given framework.

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
