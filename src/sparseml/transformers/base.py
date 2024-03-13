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


_LOGGER = logging.getLogger(__name__)


def check_transformers_install():
    try:
        import transformers  # noqa F401
    except ImportError as transformers_err:
        _LOGGER.warning(
            "transformers dependency is not installed. "
            "To install, run `pip sparseml[transformers]`"
        )
        raise transformers_err
