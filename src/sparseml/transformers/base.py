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

import logging
from typing import Optional

from sparseml.base import check_version


_LOGGER = logging.getLogger(__name__)

_TRANSFORMERS_MIN_VERSION = "4.36"
_TRANSFORMERS_MAX_VERSION = "4.37"


def check_transformers_install(
    min_version: Optional[str] = _TRANSFORMERS_MIN_VERSION,
    max_version: Optional[str] = _TRANSFORMERS_MAX_VERSION,
    raise_on_error: bool = True,
) -> bool:
    """
    Check that the transformers package is installed.
    If raise_on_error, will raise an ImportError if it is not installed or
    the required version range, if set, is not installed.
    If not raise_on_error, will return True if installed with required version
    and False otherwise.

    :param min_version: The minimum version for transformers that
        it must be greater than or equal to, if unset will require
        no minimum version
    :param max_version: The maximum version for transformers that
        it must be less than or equal to, if unset will require
        no maximum version.
    :param raise_on_error: True to raise any issues such as not installed,
        minimum version, or maximum version as ImportError.
        False to return the result.
    :return: If raise_on_error, will return False if torch is not installed
        or the version is outside the accepted bounds and True if everything
        is correct.
    """
    try:
        import transformers  # noqa F401
    except ImportError as transformers_err:
        _LOGGER.warning(
            "transformers dependency is not installed. "
            "To install, run `pip sparseml[transformers]`"
        )
        raise transformers_err

    return check_version("transformers", min_version, max_version, raise_on_error)
