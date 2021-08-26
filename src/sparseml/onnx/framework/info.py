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
support and sparsification in the ONNX/ONNXRuntime framework.
"""

import logging
from typing import Any

from sparseml.base import Framework, get_version
from sparseml.framework import FrameworkInferenceProviderInfo, FrameworkInfo
from sparseml.onnx.base import check_onnx_install, check_onnxruntime_install
from sparseml.onnx.sparsification import sparsification_info
from sparseml.sparsification import SparsificationInfo


__all__ = ["is_supported", "detect_framework", "framework_info"]


_LOGGER = logging.getLogger(__name__)


def is_supported(item: Any) -> bool:
    """
    :param item: The item to detect the support for
    :type item: Any
    :return: True if the item is supported by onnx/onnxruntime, False otherwise
    :rtype: bool
    """
    framework = detect_framework(item)

    return framework == Framework.onnx


def detect_framework(item: Any) -> Framework:
    """
    Detect the supported ML framework for a given item specifically for the
    onnx/onnxruntime package.
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
    elif isinstance(item, str) and "onnx" in item.lower().strip():
        _LOGGER.debug("framework detected from onnx text")
        # string, check if it's a string saying onnx first
        framework = Framework.onnx
    elif isinstance(item, str) and ".onnx" in item.lower().strip():
        _LOGGER.debug("framework detected from .onnx")
        # string, check if it's a file url or path that ends with onnx extension
        framework = Framework.onnx
    elif check_onnx_install(raise_on_error=False):
        from onnx import ModelProto

        if isinstance(item, ModelProto):
            _LOGGER.debug("framework detected from ONNX instance")
            # onnx native support
            framework = Framework.onnx

    return framework


def framework_info() -> FrameworkInfo:
    """
    Detect the information for the onnx/onnxruntime framework such as package versions,
    availability for core actions such as training and inference,
    sparsification support, and inference provider support.

    :return: The framework info for onnx/onnxruntime
    :rtype: FrameworkInfo
    """
    all_providers = []
    available_providers = []
    if check_onnxruntime_install(raise_on_error=False):
        from onnxruntime import get_all_providers, get_available_providers

        available_providers = get_available_providers()
        all_providers = get_all_providers()

    cpu_provider = FrameworkInferenceProviderInfo(
        name="cpu",
        description="Base CPU provider within ONNXRuntime",
        device="cpu",
        supported_sparsification=SparsificationInfo(),  # TODO: fill in when available
        available=(
            check_onnx_install(raise_on_error=False)
            and check_onnxruntime_install(raise_on_error=False)
            and "CPUExecutionProvider" in available_providers
        ),
        properties={},
        warnings=[],
    )
    gpu_provider = FrameworkInferenceProviderInfo(
        name="cuda",
        description="Base GPU CUDA provider within ONNXRuntime",
        device="gpu",
        supported_sparsification=SparsificationInfo(),  # TODO: fill in when available
        available=(
            check_onnx_install(raise_on_error=False)
            and check_onnxruntime_install(raise_on_error=False)
            and "CUDAExecutionProvider" in available_providers
        ),
        properties={},
        warnings=[],
    )

    return FrameworkInfo(
        framework=Framework.onnx,
        package_versions={
            "onnx": get_version(package_name="onnx", raise_on_error=False),
            "onnxruntime": (
                get_version(package_name="onnxruntime", raise_on_error=False)
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
        inference_providers=[cpu_provider, gpu_provider],
        properties={
            "available_providers": available_providers,
            "all_providers": all_providers,
        },
        training_available=False,
        sparsification_available=True,
        exporting_onnx_available=True,
        inference_available=True,
    )
