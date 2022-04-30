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


import functools
from typing import Optional

from sparseml.base import check_version


try:
    import torch

    torch_err = None
except Exception as err:
    torch = object()  # TODO: populate with fake object for necessary imports
    torch_err = err

try:
    import torchvision

    torchvision_err = None
except Exception as err:
    torchvision = object()  # TODO: populate with fake object for necessary imports
    torchvision_err = err


__all__ = [
    "torch",
    "torch_err",
    "torchvision",
    "torchvision_err",
    "check_torch_install",
    "check_torchvision_install",
    "require_torch",
    "require_torchvision",
]


_TORCH_MIN_VERSION = "1.0.0"
_TORCH_MAX_VERSION = "1.12.100"  # set bug to 100 to support all future 1.9.X versions


def check_torch_install(
    min_version: Optional[str] = _TORCH_MIN_VERSION,
    max_version: Optional[str] = _TORCH_MAX_VERSION,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the torch package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for torch that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for torch that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if torch is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if torch_err is not None:
        if raise_on_error:
            raise torch_err
        return False

    return check_version("torch", min_version, max_version, raise_on_error)


def check_torchvision_install(
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the torchvision package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for torchvision that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for torchvision that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if torchvision is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if torchvision_err is not None:
        if raise_on_error:
            raise torchvision_err
        return False

    return check_version("torchvision", min_version, max_version, raise_on_error)


def require_torch(
    min_version: Optional[str] = _TORCH_MIN_VERSION,
    max_version: Optional[str] = _TORCH_MAX_VERSION,
):
    """
    Decorator function to require use of torch.
    Will check that torch package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_torch_install` for more info.

    :param min_version: The minimum version for torch that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for torch that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_torch_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def require_torchvision(
    min_version: Optional[str] = None, max_version: Optional[str] = None
):
    """
    Decorator function to require use of torchvision.
    Will check that torchvision package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_torchvision_install` for more info.

    :param min_version: The minimum version for torchvision that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for torchvision that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_torchvision_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator
