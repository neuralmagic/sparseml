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


import importlib
import logging
from enum import Enum
from typing import Any, List, Optional

from packaging import version

import pkg_resources


__all__ = [
    "Framework",
    "detect_framework",
    "execute_in_sparseml_framework",
    "get_version",
    "check_version",
]


_LOGGER = logging.getLogger(__name__)


class Framework(Enum):
    """
    Framework types known of/supported within the sparseml/deepsparse ecosystem
    """

    unknown = "unknown"
    deepsparse = "deepsparse"
    onnx = "onnx"
    keras = "keras"
    pytorch = "pytorch"
    tensorflow_v1 = "tensorflow_v1"


def detect_framework(item: Any) -> Framework:
    """
    Detect the supported ML framework for a given item.
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
    _LOGGER.debug("detecting framework for %s", item)
    framework = Framework.unknown

    if isinstance(item, Framework):
        _LOGGER.debug("framework detected from Framework instance")
        framework = item
    elif isinstance(item, str) and item.lower().strip() in Framework.__members__:
        _LOGGER.debug("framework detected from Framework string instance")
        framework = Framework[item.lower().strip()]
    else:
        _LOGGER.debug("detecting framework by calling into supported frameworks")

        for test in Framework:
            try:
                framework = execute_in_sparseml_framework(
                    test, "detect_framework", item
                )
            except Exception as err:
                # errors are expected if the framework is not installed, log as debug
                _LOGGER.debug(f"error while calling detect_framework for {test}: {err}")

            if framework != Framework.unknown:
                break

    _LOGGER.info("detected framework of %s from %s", framework, item)

    return framework


def execute_in_sparseml_framework(
    framework: Framework, function_name: str, *args, **kwargs
) -> Any:
    """
    Execute a general function that is callable from the root of the frameworks
    package under SparseML such as sparseml.pytorch.
    Useful for benchmarking, analyzing, etc.
    Will pass the args and kwargs to the callable function.
    :param framework: The ML framework to run the function under in SparseML.
    :type framework: Framework
    :param function_name: The name of the function in SparseML that should be run
        with the given args and kwargs.
    :type function_name: str
    :param args: Any positional args to be passed into the function.
    :param kwargs: Any key word args to be passed into the function.
    :return: The return value from the executed function.
    :rtype: Any
    """
    _LOGGER.debug(
        "executing function with name %s for framework %s, args %s, kwargs %s",
        function_name,
        framework,
        args,
        kwargs,
    )

    if not isinstance(framework, Framework):
        framework = detect_framework(framework)

    if framework == Framework.unknown:
        raise ValueError(
            f"unknown or unsupported framework {framework}, "
            f"cannot call function {function_name}"
        )

    try:
        module = importlib.import_module(f"sparseml.{framework.value}")
        function = getattr(module, function_name)
    except Exception as err:
        raise ValueError(
            f"could not find function_name {function_name} in framework {framework}: "
            f"{err}"
        )

    return function(*args, **kwargs)


def get_version(
    package_name: str,
    raise_on_error: bool,
    alternate_package_names: Optional[List[str]] = None,
) -> Optional[str]:
    """
    :param package_name: The name of the full package, as it would be imported,
        to get the version for
    :type package_name: str
    :param raise_on_error: True to raise an error if package is not installed
        or couldn't be imported, False to return None
    :type raise_on_error: bool
    :param alternate_package_names: List of alternate names to look for the package
        under if package_name is not found. Useful for nightly builds.
    :type alternate_package_names: Optional[List[str]]
    :return: the version of the desired package if detected, otherwise raises an error
    :rtype: str
    """
    current_version: Optional[str] = None
    version_err = None

    try:
        current_version = pkg_resources.get_distribution(package_name).version
    except Exception as err:
        version_err = err

    if version_err and alternate_package_names:
        next_package = alternate_package_names.pop()

        return get_version(next_package, raise_on_error, alternate_package_names)

    if version_err and raise_on_error:
        raise ImportError(
            f"error while getting current version for {package_name}: {version_err}"
        )

    return current_version if not version_err else None


def check_version(
    package_name: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
    alternate_package_names: Optional[List[str]] = None,
) -> bool:
    """
    :param package_name: the name of the package to check the version of
    :type package_name: str
    :param min_version: The minimum version for the package that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for the package that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :param alternate_package_names: List of alternate names to look for the package
        under if package_name is not found. Useful for nightly builds.
    :type alternate_package_names: Optional[List[str]]
    :return: If raise_on_error, will return False if the package is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    current_version = get_version(package_name, raise_on_error, alternate_package_names)

    if not current_version:
        return False

    current_version = version.parse(current_version)
    min_version = version.parse(min_version) if min_version else None
    max_version = version.parse(max_version) if max_version else None

    if min_version and current_version < min_version:
        if raise_on_error:
            raise ImportError(
                f"required min {package_name} version {min_version}, "
                f"found {current_version}"
            )
        return False

    if max_version and current_version > max_version:
        if raise_on_error:
            raise ImportError(
                f"required max {package_name} version {max_version}, "
                f"found {current_version}"
            )
        return False

    return True
