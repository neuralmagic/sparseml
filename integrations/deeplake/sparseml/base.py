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
from collections import OrderedDict
from enum import Enum
from typing import Any, List, Optional

from packaging import version

import pkg_resources


__all__ = [
    "Framework",
    "detect_frameworks",
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


def _execute_sparseml_package_function(
    framework: Framework, function_name: str, *args, **kwargs
):
    try:
        module = importlib.import_module(f"sparseml.{framework.value}")
        function = getattr(module, function_name)
    except Exception as err:
        raise ValueError(
            f"unknown or unsupported framework {framework}, "
            f"cannot call function {function_name}: {err}"
        )

    return function(*args, **kwargs)


def detect_frameworks(item: Any) -> List[Framework]:
    """
    Detects the supported ML frameworks for a given item.
    Supported input types are the following:
    - A Framework enum
    - A string of any case representing the name of the framework
      (deepsparse, onnx, keras, pytorch, tensorflow_v1)
    - A supported file type within the framework such as model files:
      (onnx, pth, h5, pb)
    - An object from a supported ML framework such as a model instance
    If the framework cannot be determined, an empty list will be returned

    :param item: The item to detect the ML framework for
    :type item: Any
    :return: The detected ML frameworks from the given item
    :rtype: List[Framework]
    """
    _LOGGER.debug("detecting frameworks for %s", item)
    frameworks = []

    if isinstance(item, str) and item.lower().strip() in Framework.__members__:
        _LOGGER.debug("framework detected from Framework string instance")
        item = Framework[item.lower().strip()]

    if isinstance(item, Framework):
        _LOGGER.debug("framework detected from Framework instance")

        if item != Framework.unknown:
            frameworks.append(item)
    else:
        _LOGGER.debug("detecting frameworks by calling into supported frameworks")
        frameworks = []

        for test in Framework:
            if test == Framework.unknown:
                continue

            try:
                detected = _execute_sparseml_package_function(
                    test, "detect_framework", item
                )
                if detected != Framework.unknown:
                    frameworks.append(detected)
            except Exception as err:
                # errors are expected if the framework is not installed, log as debug
                _LOGGER.debug(
                    "error while calling detect_framework for %s: %s", test, err
                )

    _LOGGER.info("detected frameworks of %s from %s", frameworks, item)

    return frameworks


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
    frameworks = detect_frameworks(item)

    return frameworks[0] if len(frameworks) > 0 else Framework.unknown


def execute_in_sparseml_framework(
    framework: Any, function_name: str, *args, **kwargs
) -> Any:
    """
    Execute a general function that is callable from the root of the frameworks
    package under SparseML such as sparseml.pytorch.
    Useful for benchmarking, analyzing, etc.
    Will pass the args and kwargs to the callable function.

    :param framework: The item to detect the ML framework for to run the function under,
        see detect_frameworks for more details on acceptible inputs
    :type framework: Any
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

    framework_errs = OrderedDict()
    test_frameworks = detect_frameworks(framework)

    for test_framework in test_frameworks:
        try:
            module = importlib.import_module(f"sparseml.{test_framework.value}")
            function = getattr(module, function_name)

            return function(*args, **kwargs)
        except Exception as err:
            framework_errs[framework] = err

    if len(framework_errs) == 1:
        raise list(framework_errs.values())[0]

    if len(framework_errs) > 1:
        raise RuntimeError(str(framework_errs))

    raise ValueError(
        f"unknown or unsupported framework {framework}, "
        f"cannot call function {function_name}"
    )


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
    extra_error_message: Optional[str] = None,
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
    :param extra_error_message: optional string to append to error message if error
        is raised
    :type extra_error_message: Optional[str]
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
                + (f". {extra_error_message}" if extra_error_message else "")
            )
        return False

    if max_version and current_version > max_version:
        if raise_on_error:
            raise ImportError(
                f"required max {package_name} version {max_version}, "
                f"found {current_version}"
                + (f". {extra_error_message}" if extra_error_message else "")
            )
        return False

    return True
