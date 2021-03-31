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
Functions and classes for detecting and working with functionality
for models within ML frameworks.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from sparseml.base import Framework, detect_framework, execute_in_sparseml_framework
from sparseml.sparsification.info import SparsificationInfo


__all__ = [
    "FrameworkInferenceProviderInfo",
    "FrameworkInfo",
    "framework_info",
]


_LOGGER = logging.getLogger(__name__)


class FrameworkInferenceProviderInfo(BaseModel):
    """
    Class for storing information for an inference provider within a frameworks engine.
    For example, the gpu provider within PyTorch.

    Extends pydantics BaseModel class for serialization to and from json
    in addition to proper type checking on construction.
    """

    name: str = Field(title="name", description="The name/id of the provider.")
    description: str = Field(
        title="description",
        description="A description for the provider to offer more detail.",
    )
    device: str = Field(
        title="device", description="The device the provider is for such as cpu or gpu."
    )
    supported_sparsification: SparsificationInfo = Field(
        title="supported_sparsification",
        description=(
            "The supported sparsification information for support "
            "for inference speedup in the provider."
        ),
    )

    available: bool = Field(
        default=False,
        title="available",
        description="True if the provider is available on the system, False otherwise.",
    )
    properties: Dict[str, Any] = Field(
        default=OrderedDict(),
        title="properties",
        description="Any properties for the given provider.",
    )
    warnings: List[str] = Field(
        default=[],
        title="warnings",
        description="Any warnings/restrictions for the provider on the given system.",
    )


class FrameworkInfo(BaseModel):
    """
    Class for storing the information for an ML frameworks info and availability
    on the current system.

    Extends pydantics BaseModel class for serialization to and from json
    in addition to proper type checking on construction.
    """

    framework: Framework = Field(
        title="framework", description="The framework the system info is for."
    )
    package_versions: Dict[str, str] = Field(
        title="package_versions",
        description=(
            "A mapping of the package and supporting packages for a given framework "
            "to the detected versions on the system currently. "
            "If the package is not detected, will be set to 'not detected'."
        ),
    )
    sparsification: SparsificationInfo = Field(
        title="sparsification",
        description=(
            "True if inference for a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )
    inference_providers: List[FrameworkInferenceProviderInfo] = Field(
        title="inference_providers",
        description=(
            "True if inference for a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )

    training_available: bool = Field(
        default=False,
        title="training_available",
        description=(
            "True if training/editing a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )
    sparsification_available: bool = Field(
        default=False,
        title="sparsification_available",
        description=(
            "True if sparsifying a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )
    exporting_onnx_available: bool = Field(
        default=False,
        title="exporting_onnx_available",
        description=(
            "True if exporting a model in the ONNX format is available on the system "
            "for the given framework, False otherwise."
        ),
    )
    inference_available: bool = Field(
        default=False,
        title="inference_available",
        description=(
            "True if inference for a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )


def framework_info(framework: Any) -> FrameworkInfo:
    """
    Detect the information for the given ML framework such as package versions,
    availability for core actions such as training and inference,
    sparsification support, and inference provider support.

    :param framework: The item to detect the ML framework for.
        See :func:`detect_framework` for more information.
    :type framework: Any
    :return: The framework info for the given framework
    :rtype: FrameworkInfo
    """
    _LOGGER.debug("getting system info for framework %s", framework)
    info: FrameworkInfo = execute_in_sparseml_framework(framework, "framework_info")
    _LOGGER.info("retrieved system info for framework %s: %s", framework, info)

    return info
