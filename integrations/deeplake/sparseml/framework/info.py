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
Functionality related to integrating with, detecting, and getting information for
support and sparsification in ML frameworks.

The file is executable and will get the framework info for a given framework:

##########
Command help:
usage: info.py [-h] [--path PATH] framework

Compile the available setup and information for a given framework.

positional arguments:
  framework    the ML framework or path to a framework file to load the
               framework info for

optional arguments:
  -h, --help   show this help message and exit
  --path PATH  A full file path to save the framework info to. If not
               supplied, will print out the framework info to the
               console.

#########
EXAMPLES
#########

##########
Example command for getting the framework info for pytorch.
python src/sparseml/framework/info.py pytorch
"""

import argparse
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sparseml.base import Framework, execute_in_sparseml_framework
from sparseml.sparsification.info import SparsificationInfo
from sparseml.utils import clean_path, create_parent_dirs


__all__ = [
    "FrameworkInferenceProviderInfo",
    "FrameworkInfo",
    "framework_info",
    "save_framework_info",
    "load_framework_info",
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
    supported_sparsification: Optional[SparsificationInfo] = Field(
        default=None,
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
    package_versions: Dict[str, Optional[str]] = Field(
        title="package_versions",
        description=(
            "A mapping of the package and supporting packages for a given framework "
            "to the detected versions on the system currently. "
            "If the package is not detected, will be set to None."
        ),
    )
    sparsification: Optional[SparsificationInfo] = Field(
        default=None,
        title="sparsification",
        description=(
            "True if inference for a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )
    inference_providers: List[FrameworkInferenceProviderInfo] = Field(
        default=[],
        title="inference_providers",
        description=(
            "True if inference for a model is available on the system "
            "for the given framework, False otherwise."
        ),
    )
    properties: Dict[str, Any] = Field(
        default={},
        title="properties",
        description="Any additional properties for the framework.",
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


def save_framework_info(framework: Any, path: Optional[str] = None):
    """
    Save the framework info for a given framework.
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
        "saving framework info for framework %s to %s",
        framework,
        path if path else "sys.out",
    )
    info = (
        framework_info(framework)
        if not isinstance(framework, FrameworkInfo)
        else framework
    )

    if path:
        path = clean_path(path)
        create_parent_dirs(path)

        with open(path, "w") as file:
            file.write(info.json())

        _LOGGER.info(
            "saved framework info for framework %s in file at %s", framework, path
        ),
    else:
        print(info.json(indent=4))
        _LOGGER.info("printed out framework info for framework %s", framework)


def load_framework_info(load: str) -> FrameworkInfo:
    """
    Load the framework info from a file or raw json.
    If load exists as a path, will read from the file and use that.
    Otherwise will try to parse the input as a raw json str.

    :param load: Either a file path to a json file or a raw json string.
    :type load: str
    :return: The loaded framework info.
    :rtype: FrameworkInfo
    """
    loaded_path = clean_path(load)

    if os.path.exists(loaded_path):
        with open(loaded_path, "r") as file:
            load = file.read()

    info = FrameworkInfo.parse_raw(load)

    return info


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compile the available setup and information for a given framework."
        )
    )
    parser.add_argument(
        "framework",
        type=str,
        help=(
            "the ML framework or path to a framework file to load the "
            "framework info for"
        ),
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "A full file path to save the framework info to. "
            "If not supplied, will print out the framework info to the console."
        ),
    )

    return parser.parse_args()


def _main():
    args = _parse_args()
    save_framework_info(args.framework, args.path)


if __name__ == "__main__":
    _main()
