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

import pytest

from tests.integrations.base_tester import BaseIntegrationTester
from yolov5.export import create_checkpoint, load_checkpoint
from yolov5.val import run as val

from .yolov5_args import Yolov5TrainArgs


@pytest.mark.usefixtures("setup")
class Yolov5IntegrationTester(BaseIntegrationTester):
    command_stubs = {
        "train": "sparseml.yolov5.train",
        "export": "sparseml.yolov5.export",
        "deploy": "sparseml.yolov5.deploy",
    }
    command_args_classes = {
        "train": Yolov5TrainArgs,
    }

    def test_checkpoint_load(self, setup):
        pass
