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
Functionality related to describing availability and information of sparsification
algorithms to models within in the DeepSparse framework.
"""

import logging

from sparseml.sparsification import SparsificationInfo


__all__ = ["sparsification_info"]


_LOGGER = logging.getLogger(__name__)


def sparsification_info() -> SparsificationInfo:
    """
    Load the available setup for sparsifying model within deepsparse.
    :return: The sparsification info for the deepsparse framework
    :rtype: SparsificationInfo
    """
    _LOGGER.debug("getting sparsification info for deepsparse")
    info = SparsificationInfo(modifiers=[])
    _LOGGER.info("retrieved sparsification info for deepsparse: %s", info)

    return info
