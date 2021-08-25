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
Functionality related to detecting and getting information for
support and sparsification in the DeepSparse framework.
"""

import logging
from typing import Any

from sparseml.base import Framework, get_version
from sparseml.deepsparse.base import check_deepsparse_install
from sparseml.deepsparse.sparsification import sparsification_info
from sparseml.framework import FrameworkInferenceProviderInfo, FrameworkInfo
from sparseml.sparsification import SparsificationInfo
from sparsezoo import File, Model


__all__ = ["is_supported", "detect_framework", "framework_info"]


_LOGGER = logging.getLogger(__name__)


def is_supported(item: Any) -> bool:
    """
    :param item: The item to detect the support for
    :type item: Any
    :return: True if the item is supported by deepsparse, False otherwise
    :rtype: bool
    """
    framework = detect_framework(item)

    return framework == Framework.deepsparse


def detect_framework(item: Any) -> Framework:
    """
    Detect the supported ML framework for a given item specifically for the
    deepsparse package.
    Supported input types are the following:
    - A Framework enum
    - A string of any case representing the name of the framework
      (deepsparse, onnx, keras, pytorch, tensorflow_v1)
    - A supported file type within the framework such as model files:
      (onnx, pth, h5, pb)
    - An object from a supported ML framework such as a model instance
    If the framework cannot be determined, will return Framework.unknown

    :param item: The item to detect the ML framework for
    :type item: Any
    :return: The detected framework from the given item
    :rtype: Framework
    """
    framework = Framework.unknown

    if isinstance(item, Framework):
        _LOGGER.debug("framework detected from Framework instance")
        framework = item
    elif isinstance(item, str) and item.lower().strip() in Framework.__members__:
        _LOGGER.debug("framework detected from Framework string instance")
        framework = Framework[item.lower().strip()]
    elif isinstance(item, str) and (
        "deepsparse" in item.lower().strip() or "deep sparse" in item.lower().strip()
    ):
        _LOGGER.debug("framework detected from deepsparse text")
        # string, check if it's a string saying deepsparse first
        framework = Framework.deepsparse
    elif isinstance(item, str) and ".onnx" in item.lower().strip():
        _LOGGER.debug("framework detected from .onnx")
        # string, check if it's a file url or path that ends with onnx extension
        framework = Framework.deepsparse
    elif isinstance(item, Model) or isinstance(item, File):
        _LOGGER.debug("framework detected from SparseZoo instance")
        # sparsezoo model/file, deepsparse supports these natively
        framework = Framework.deepsparse

    return framework


def framework_info() -> FrameworkInfo:
    """
    Detect the information for the deepsparse framework such as package versions,
    availability for core actions such as training and inference,
    sparsification support, and inference provider support.

    :return: The framework info for deepsparse
    :rtype: FrameworkInfo
    """
    arch = {}

    if check_deepsparse_install(raise_on_error=False):
        from deepsparse.cpu import cpu_architecture

        arch = cpu_architecture()

    cpu_warnings = []
    if arch and arch.isa != "avx512":
        cpu_warnings.append(
            "AVX512 instruction set not detected, inference performance will be limited"
        )
    if arch and arch.isa != "avx512" and arch.isa != "avx2":
        cpu_warnings.append(
            "AVX2 and AVX512 instruction sets not detected, "
            "inference performance will be severely limited"
        )
    if arch and not arch.vnni:
        cpu_warnings.append(
            "VNNI instruction set not detected, "
            "quantized inference performance will be limited"
        )

    cpu_provider = FrameworkInferenceProviderInfo(
        name="cpu",
        description=(
            "Performant CPU provider within DeepSparse specializing in speedup of "
            "sparsified models using AVX and VNNI instruction sets"
        ),
        device="cpu",
        supported_sparsification=SparsificationInfo(),  # TODO: fill in when available
        available=check_deepsparse_install(raise_on_error=False),
        properties={
            "cpu_architecture": arch,
        },
        warnings=cpu_warnings,
    )

    return FrameworkInfo(
        framework=Framework.deepsparse,
        package_versions={
            "deepsparse": get_version(
                package_name="deepsparse",
                raise_on_error=False,
                alternate_package_names=["deepsparse-nightly"],
            ),
            "sparsezoo": get_version(
                package_name="sparsezoo",
                raise_on_error=False,
                alternate_package_names=["sparsezoo-nightly"],
            ),
            "sparseml": get_version(
                package_name="sparseml",
                raise_on_error=False,
                alternate_package_names=["sparseml-nightly"],
            ),
        },
        sparsification=sparsification_info(),
        inference_providers=[cpu_provider],
        training_available=False,
        sparsification_available=False,
        exporting_onnx_available=False,
        inference_available=True,
    )
