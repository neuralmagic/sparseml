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
    import keras

    keras_err = None
    is_native_keras = True
except Exception as err:
    keras = object()
    keras_err = err
    is_native_keras = False

try:
    import tensorflow

    tensorflow_err = None

    if keras_err:
        from tensorflow import keras

        keras_err = None
except Exception as err:
    tensorflow = object()  # TODO: populate with fake object for necessary improvements
    tensorflow_err = err


try:
    import keras2onnx

    keras2onnx_err = None
except Exception as err:
    keras2onnx = object()  # TODO: populate with fake object for necessary imports
    keras2onnx_err = err


__all__ = [
    "keras",
    "keras_err",
    "tensorflow",
    "tensorflow_err",
    "keras2onnx",
    "keras2onnx_err",
    "is_native_keras",
    "check_keras_install",
    "check_keras2onnx_install",
    "require_keras",
    "require_keras2onnx",
]

_DEF_TF_MIN_VERSION = "2.1.0"
_DEF_KERAS_MIN_VERSION = "2.4.3"
_KERAS2ONNX_MIN_VERSION = "1.0.0"


def check_keras_install(
    min_tf_version: Optional[str] = _DEF_TF_MIN_VERSION,
    max_tf_version: Optional[str] = None,
    min_native_version: Optional[str] = _DEF_KERAS_MIN_VERSION,
    require_tensorflow_backend: bool = True,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the keras package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_tf_version: The minimum version for keras that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_tf_version: str
    :param max_tf_version: The maximum version for keras that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_tf_version: str
    :param min_native_version: The minimum version for native keras that it must be
        greater than or equal to if installed
    :type min_native_version: str
    :param require_tensorflow_backend: True to require keras to use the tensorflow
        backend, False otherwise.
    :type require_tensorflow_backend: bool
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if keras is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if keras_err is not None:
        if raise_on_error:
            raise keras_err
        return False

    if tensorflow_err is not None and require_tensorflow_backend:
        if raise_on_error:
            raise tensorflow_err
        return False

    if require_tensorflow_backend and not check_version(
        "tensorflow", min_tf_version, max_tf_version, raise_on_error
    ):
        return False

    if is_native_keras and not check_version(
        "keras", min_native_version, None, raise_on_error
    ):
        return False

    return True


def check_keras2onnx_install(
    min_version: Optional[str] = _KERAS2ONNX_MIN_VERSION,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the keras2onnx package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for keras2onnx that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for keras2onnx that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if keras2onnx is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if keras2onnx_err is not None:
        if raise_on_error:
            raise keras2onnx_err
        return False

    return check_version("keras2onnx", min_version, max_version, raise_on_error)


def require_keras(
    min_tf_version: Optional[str] = _DEF_TF_MIN_VERSION,
    max_tf_version: Optional[str] = None,
    min_native_version: Optional[str] = _DEF_KERAS_MIN_VERSION,
    require_tensorflow_backend: bool = True,
):
    """
    Decorator function to require use of keras.
    Will check that keras package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_keras_install` for more info.

    :param min_tf_version: The minimum version for keras that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_tf_version: str
    :param max_tf_version: The maximum version for keras that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_tf_version: str
    :param min_native_version: The minimum version for native keras that it must be
        greater than or equal to if installed
    :type min_native_version: str
    :param require_tensorflow_backend: True to require keras to use the tensorflow
        backend, False otherwise.
    :type require_tensorflow_backend: bool
    :param require_tensorflow_backend: True to require keras to use the tensorflow
        backend, False otherwise.
    :type require_tensorflow_backend: bool
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_keras_install(
                min_tf_version,
                max_tf_version,
                min_native_version,
                require_tensorflow_backend,
            )

            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def require_keras2onnx(
    min_version: Optional[str] = _KERAS2ONNX_MIN_VERSION,
    max_version: Optional[str] = None,
):
    """
    Decorator function to require use of keras2onnx.
    Will check that keras2onnx package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_keras2onnx_install` for more info.

    param min_version: The minimum version for keras2onnx that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for keras2onnx that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_keras2onnx_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator
