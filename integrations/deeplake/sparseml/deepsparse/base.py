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
    import deepsparse

    deepsparse_err = None
except Exception as err:
    deepsparse = object()  # TODO: populate with fake object for necessary imports
    deepsparse_err = err


__all__ = [
    "deepsparse",
    "deepsparse_err",
    "check_deepsparse_install",
    "require_deepsparse",
]


def check_deepsparse_install(
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the deepsparse package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for deepsparse that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for deepsparse that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError. False to return the result.
    :type raise_on_error: bool
    :return: If raise_on_error, will return False if deepsparse is not installed
        or the version is outside the accepted bounds and True if everything is correct.
    :rtype: bool
    """
    if deepsparse_err is not None:
        if raise_on_error:
            raise deepsparse_err
        return False

    return check_version(
        "deepsparse",
        min_version,
        max_version,
        raise_on_error,
        alternate_package_names=["deepsparse-nightly", "deepsparse-ent"],
    )


def require_deepsparse(
    min_version: Optional[str] = None, max_version: Optional[str] = None
):
    """
    Decorator function to require use of deepsparse.
    Will check that deepsparse package is installed and within the bounding
    ranges of min_version and max_version if they are set before calling
    the wrapped function.
    See :func:`check_deepsparse_install` for more info.

    param min_version: The minimum version for deepsparse that it must be greater than
        or equal to, if unset will require no minimum version
    :type min_version: str
    :param max_version: The maximum version for deepsparse that it must be less than
        or equal to, if unset will require no maximum version.
    :type max_version: str
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            check_deepsparse_install(min_version, max_version)

            return func(*args, **kwargs)

        return _wrapper

    return _decorator
