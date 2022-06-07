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
    import onnx

    onnx_err = None
except Exception as err:
    onnx = object()  # TODO: populate with fake object for necessary imports
    onnx_err = err

try:
    import onnxruntime

    onnxruntime_err = None

except Exception as error:
    onnxruntime = object()  # TODO: populate with fake object for necessary imports
    onnxruntime_err = error

__all__ = [
    "onnx",
    "onnx_err",
    "onnxruntime",
    "onnxruntime_err",
    "check_onnx_install",
    "check_onnxruntime_install",
    "require_onnx",
    "require_onnxruntime",
]


_ONNX_MIN_VERSION = "1.5.0"
_ORT_MIN_VERSION = "1.0.0"


def check_onnx_install(
    min_version: Optional[str] = _ONNX_MIN_VERSION,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the onnx package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for onnx that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for onnx that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if onnx is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if onnx_err is not None:
        if raise_on_error:
            raise onnx_err
        return False

    return check_version("onnx", min_version, max_version, raise_on_error)


def check_onnxruntime_install(
    min_version: Optional[str] = _ORT_MIN_VERSION,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the onnxruntime package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for onnxruntime that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for onnxruntime that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if onnxruntime is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if onnxruntime_err is not None:
        if raise_on_error:
            raise onnxruntime_err
        return False

    return check_version(
        "onnxruntime",
        min_version,
        max_version,
        raise_on_error,
        extra_error_message="Try installing sparseml[onnxruntime] or onnxruntime",
    )


def require_onnx(
    min_version: Optional[str] = _ONNX_MIN_VERSION, max_version: Optional[str] = None
):
    """
    Decorator function to require use of onnx.
    Will check that onnx package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_onnx_install` for more info.

    param min_version: The minimum version for onnx that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for onnx that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_onnx_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def require_onnxruntime(
    min_version: Optional[str] = _ORT_MIN_VERSION, max_version: Optional[str] = None
):
    """
    Decorator function to require use of onnxruntime.
    Will check that onnxruntime package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_onnxruntime_install` for more info.

    param min_version: The minimum version for onnxruntime that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for onnxruntime that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_onnxruntime_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator
