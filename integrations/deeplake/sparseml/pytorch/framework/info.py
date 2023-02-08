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
support and sparsification in the PyTorch framework.
"""

import logging
from typing import Any

from sparseml.base import Framework, get_version
from sparseml.framework import FrameworkInferenceProviderInfo, FrameworkInfo
from sparseml.pytorch.base import check_torch_install, torch
from sparseml.pytorch.sparsification import sparsification_info
from sparseml.sparsification import SparsificationInfo


__all__ = ["is_supported", "detect_framework", "framework_info"]


_LOGGER = logging.getLogger(__name__)


def is_supported(item: Any) -> bool:
    """
    :param item: The item to detect the support for
    :type item: Any
    :return: True if the item is supported by pytorch, False otherwise
    :rtype: bool
    """
    framework = detect_framework(item)

    return framework == Framework.pytorch


def detect_framework(item: Any) -> Framework:
    """
    Detect the supported ML framework for a given item specifically for the
    pytorch package.
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
    elif isinstance(item, str) and "torch" in item.lower().strip():
        _LOGGER.debug("framework detected from torch text")
        # string, check if it's a string saying onnx first
        framework = Framework.pytorch
    elif isinstance(item, str) and (
        ".pt" in item.lower().strip() or ".mar" in item.lower().strip()
    ):
        _LOGGER.debug("framework detected from .pt or .mar")
        # string, check if it's a file url or path that ends with onnx extension
        framework = Framework.pytorch
    elif check_torch_install(raise_on_error=False):
        from torch.nn import Module

        if isinstance(item, Module):
            _LOGGER.debug("framework detected from pytorch instance")
            # pytorch native support
            framework = Framework.pytorch

    return framework


def framework_info() -> FrameworkInfo:
    """
    Detect the information for the onnx/onnxruntime framework such as package versions,
    availability for core actions such as training and inference,
    sparsification support, and inference provider support.

    :return: The framework info for onnx/onnxruntime
    :rtype: FrameworkInfo
    """
    cpu_provider = FrameworkInferenceProviderInfo(
        name="cpu",
        description="Base CPU provider within PyTorch",
        device="cpu",
        supported_sparsification=SparsificationInfo(),  # TODO: fill in when available
        available=check_torch_install(raise_on_error=False),
        properties={},
        warnings=[],
    )
    gpu_provider = FrameworkInferenceProviderInfo(
        name="cuda",
        description="Base GPU CUDA provider within PyTorch",
        device="gpu",
        supported_sparsification=SparsificationInfo(),  # TODO: fill in when available
        available=(
            check_torch_install(raise_on_error=False) and torch.cuda.is_available()
        ),
        properties={},
        warnings=[],
    )

    return FrameworkInfo(
        framework=Framework.pytorch,
        package_versions={
            "torch": get_version(package_name="torch", raise_on_error=False),
            "torchvision": (
                get_version(package_name="torchvision", raise_on_error=False)
            ),
            "onnx": get_version(package_name="onnx", raise_on_error=False),
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
        properties={},
        training_available=True,
        sparsification_available=True,
        exporting_onnx_available=True,
        inference_available=True,
    )
