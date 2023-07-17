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
Functionality for working with and sparsifying Models in the Keras framework
"""

# flake8: noqa

from sparseml.utils import deprecation_warning as _deprecation_warning


_deprecation_warning(
    "sparseml.keras is deprecated and will be removed in a future version",
)

from sparseml.analytics import sparseml_analytics as _analytics

from .base import *
from .framework import detect_framework, framework_info, is_supported
from .sparsification import sparsification_info


_analytics.send_event("python__keras__init")
