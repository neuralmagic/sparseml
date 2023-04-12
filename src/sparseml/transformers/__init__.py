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
Tools for integrating SparseML with transformers training flows
"""

# flake8: noqa

import logging as _logging

from sparseml.analytics import sparseml_analytics as _analytics


_analytics.send_event("python__transformers__init")


_LOGGER = _logging.getLogger(__name__)


def _check_transformers_install():
    # check for NM integration in transformers version
    import transformers as _transformers

    if not _transformers.NM_INTEGRATED:
        _LOGGER.warning(
            "the neuralmagic fork of transformers may not be installed. it can be "
            "installed via "
            f"`pip install {_NM_TRANSFORMERS_NIGHTLY}`"
        )


_check_transformers_install()

from .export import *
