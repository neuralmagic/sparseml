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

import os
import tempfile
from collections import OrderedDict

import onnx
import onnxruntime as ort
import pytest
import torch

from tests.integrations.base_tester import (
    BaseIntegrationManager,
    BaseIntegrationTester,
    skip_inactive_stage,
)
from tests.integrations.helpers import get_configs_with_cadence
from tests.integrations.transformers.transformers_args import (
    TransformersTrainArgs,
)


try:
    import deepsparse
except Exception:
    deepsparse = None


METRIC_TO_INDEX = {}


class TransformersManager(BaseIntegrationManager):
    command_stubs = {
        "train": "sparseml.transformers{task}.train",
        "export": "sparseml.transformers{task}.export"
    }
    config_classes = {
        "train": TransformersTrainArgs
    }