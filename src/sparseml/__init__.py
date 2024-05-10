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
Tooling to help train, test, and optimize models for better performance
"""

# flake8: noqa
# isort: skip_file

# be sure to import all logging first and at the root
# this keeps other loggers in nested files creating from the root logger setups
from .log import *
from .version import *

from .core import *
from .base import check_version, detect_framework, execute_in_sparseml_framework
from .framework import (
    FrameworkInferenceProviderInfo,
    FrameworkInfo,
    framework_info,
    save_framework_info,
    load_framework_info,
)
from .sparsification import (
    SparsificationInfo,
    sparsification_info,
    save_sparsification_info,
    load_sparsification_info,
)
from .analytics import sparseml_analytics as _analytics
from .export.export import export
from .evaluation.evaluator import evaluate

_analytics.send_event("python__init")
