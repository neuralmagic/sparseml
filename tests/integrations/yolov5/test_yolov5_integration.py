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
from collections import Counter
from pathlib import Path
import tempfile

import onnx
import pandas as pd
import pytest

from tests.integrations.base_tester import BaseIntegrationTester
from tests.integrations.helpers import get_configs_with_cadence, skip_inactive_stage
from tests.integrations.yolov5.yolov5_args import Yolov5ExportArgs, Yolov5TrainArgs
from yolov5.utils.general import ROOT, increment_path
from yolov5.val import run as val


METRIC_TO_INDEX = {"map0.5": 2}


@pytest.mark.parametrize(
    "setup",
    get_configs_with_cadence(
        os.environ.get("NM_TEST_CADENCE", "commit"), os.path.dirname(__file__)
    ),
    indirect=True,
    scope="class",
)
# Iterate over configs with the matching cadence (default commit)
class TestYolov5Integration(BaseIntegrationTester):

    command_stubs = {
        "train": "sparseml.object_detection.train",
        "export": "sparseml.object_detection.export",
        "deploy": "sparseml.object_detection.deploy",
    }
    command_args_classes = {
        "train": Yolov5TrainArgs,
        "export": Yolov5ExportArgs,
    }

    def capture_pre_run_state(self):
        args = self.configs["train"]["args"]
        self.save_dir = tempfile.TemporaryDirectory(dir=args.project)
        args.project = self.save_dir.name

        if "export" in self.command_types:
            self.configs["export"]["args"].weights = os.path.join(
                args.project, "exp", "weights", "last.pt"
            )

    def teardown(self):
        #self.save_dir.cleanup()
        pass

@skip_inactive_stage
def test_train_val(self):
    args = self.configs["train"]["args"]
    model_file = os.path.join(self.save_dir.name, "exp", "weights", "last.pt")
    assert os.path.isfile(model_file)
    metrics, *_ = val(data=ROOT / "data/coco128.yaml", weights=model_file)
    if "target_name" in self.test_args["train"]:
        test_args = self.test_args["train"]
        metric_idx = METRIC_TO_INDEX[test_args["target_name"]]
        metric = metrics[metric_idx]
        target_mean = test_args["target_mean"]
        target_std = test_args["target_std"]
        assert target_mean - target_std <= metric <= target_mean + target_std

@skip_inactive_stage
def test_export_onnx_graph(self):
    model = onnx.load(
        os.path.join(os.path.dirname(self.configs["export"]["args"].weights), "last.onnx")
    )
