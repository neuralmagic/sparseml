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
import os
from typing import Optional

from sparseml.base import check_version


try:
    import tensorflow

    tf_compat = (
        tensorflow
        if not hasattr(tensorflow, "compat")
        or not hasattr(getattr(tensorflow, "compat"), "v1")
        else tensorflow.compat.v1
    )
    tensorflow_err = None
except Exception as err:
    tensorflow = object()  # TODO: populate with fake object for necessary imports
    tf_compat = object()  # TODO: populate with fake object for necessary imports
    tensorflow_err = err


try:
    import tf2onnx

    tf2onnx_err = None
except Exception as err:
    tf2onnx = object()  # TODO: populate with fake object for necessary imports
    tf2onnx_err = err


__all__ = [
    "tensorflow",
    "tf_compat",
    "tensorflow_err",
    "tf2onnx",
    "tf2onnx_err",
    "check_tensorflow_install",
    "check_tf2onnx_install",
    "require_tensorflow",
    "require_tf2onnx",
]


_TENSORFLOW_MIN_VERSION = "1.8.0"
_TENSORFLOW_MAX_VERSION = "1.16.0"

_TF2ONNX_MIN_VERSION = "1.0.0"


def check_tensorflow_install(
    min_version: Optional[str] = _TENSORFLOW_MIN_VERSION,
    max_version: Optional[str] = _TENSORFLOW_MAX_VERSION,
    raise_on_error: bool = True,
    allow_env_ignore_flag: bool = True,
) -> bool:
    """
    Check that the tensorflow package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for tensorflow that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for tensorflow that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :param allow_env_ignore_flag: True to allow the env variable SPARSEML_IGNORE_TFV1
        to ignore the tensorflow install and version checks.
        False to ignore the ignore flag.
    :type allow_env_ignore_flag: bool
    :return: If raise_on_error, will return False if tensorflow is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if allow_env_ignore_flag and os.getenv("SPARSEML_IGNORE_TFV1", False):
        return True

    if tensorflow_err is not None:
        if raise_on_error:
            raise tensorflow_err
        return False

    return check_version(
        "tensorflow",
        min_version,
        max_version,
        raise_on_error,
        alternate_package_names=["tensorflow-gpu"],
    )


def check_tf2onnx_install(
    min_version: Optional[str] = _TF2ONNX_MIN_VERSION,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the tf2onnx package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for tf2onnx that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for tf2onnx that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if tf2onnx is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if tf2onnx_err is not None:
        if raise_on_error:
            raise tf2onnx_err
        return False

    return check_version("tf2onnx", min_version, max_version, raise_on_error)


def require_tensorflow(
    min_version: Optional[str] = _TENSORFLOW_MIN_VERSION,
    max_version: Optional[str] = _TENSORFLOW_MAX_VERSION,
    allow_env_ignore_flag: bool = True,
):
    """
    Decorator function to require use of tensorflow.
    Will check that tensorflow package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_tensorflow_install` for more info.

    :param min_version: The minimum version for tensorflow that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for tensorflow that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param allow_env_ignore_flag: True to allow the env variable SPARSEML_IGNORE_TFV1
        to ignore the tensorflow install and version checks.
        False to ignore the ignore flag.
    :type allow_env_ignore_flag: bool
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_tensorflow_install(min_version, max_version, allow_env_ignore_flag)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def require_tf2onnx(
    min_version: Optional[str] = _TF2ONNX_MIN_VERSION,
    max_version: Optional[str] = None,
):
    """
    Decorator function to require use of tf2onnx.
    Will check that tf2onnx package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_tf2onnx_install` for more info.

    :param min_version: The minimum version for tf2onnx that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for tf2onnx that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_tf2onnx_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator
