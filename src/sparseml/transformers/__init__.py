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


try:
    import datasets as _datasets
    import transformers as _transformers
except ImportError:
    raise ImportError("Please install sparseml[transformers] to use this pathway")


_analytics.send_event("python__transformers__init")


_LOGGER = _logging.getLogger(__name__)


def _check_transformers_install():
    # check for NM integration in transformers version
    import transformers as _transformers

    if not getattr(_transformers, "NM_INTEGRATED", False):
        message = (
            "****************************************************************\n"
            "WARNING: It appears that the Neural Magic fork of Transformers is not installed!\n"
            "This is CRITICAL for the proper application of quantization in SparseML flows.\n\n"
            "To resolve this, please run: `pip uninstall transformers;pip install nm-transformers`\n"
            "Failing to do so is UNSUPPORTED and may significantly affect model performance.\n"
            "****************************************************************"
        )
        _LOGGER.warning(message)


_check_transformers_install()

from .export import *
