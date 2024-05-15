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

from packaging import version

from sparseml.base import get_version


_LOGGER = logging.getLogger(__name__)

__all__ = ["check_transformers_version"]

_TRANSFORMERS_MIN_VERSION = "4.40.0"
_TRANSFORMERS_MAX_VERSION = "4.41.0"


def check_transformers_version(
    module_version_max: str = _TRANSFORMERS_MAX_VERSION,
    model_version_min: str = _TRANSFORMERS_MIN_VERSION,
) -> bool:
    """
    Checks whether the transformers package version installed falls within
    the range specified by the module_version_max and model_version_min parameters.
    If not, a warning is logged.

    :param module_version_max: The maximum version for transformers that allows this
        function to return True.
    :param model_version_min: The minimum version for transformers that allows this
        function to return True.
    :return: True if the transformers package version is within the specified range,
        False otherwise.
    """
    current_version = get_version(package_name="transformers", raise_on_error=True)

    if not current_version:
        return False

    current_version = version.parse(current_version)
    max_version = version.parse(module_version_max)
    min_version = version.parse(model_version_min)

    if not (min_version <= current_version <= max_version):
        _LOGGER.warning(
            "Attempting to modify the transformers model to support "
            "the SparseML-specific functionalities. However, the detected "
            f"transformers version ({current_version}) does not fall within the "
            f"supported version range ({min_version} - {max_version}). "
            "This may lead to unexpected behavior. Please ensure that the "
            "correct transformers version is installed."
        )
        return False
    return True
