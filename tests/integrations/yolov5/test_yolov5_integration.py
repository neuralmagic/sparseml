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

import onnx
import pandas as pd
import pytest

from tests.integrations.base_tester import BaseIntegrationTester
from tests.integrations.helpers import get_configs_with_cadence, skip_inactive_stage
from tests.integrations.yolov5.yolov5_args import Yolov5TrainArgs
from yolov5.export import create_checkpoint, load_checkpoint
from yolov5.val import run as val


pytest.mark.parametrize(
    "config_path",
    ["test"],  # get_configs_with_cadence(os.environ.get("NM_TEST_CADENCE")),
)


@pytest.mark.usefixtures("setup")
class TestYolov5Integration(BaseIntegrationTester):

    command_stubs = {
        "train": "sparseml.yolov5.train",
        "export": "sparseml.yolov5.export",
        "deploy": "sparseml.yolov5.deploy",
    }
    command_args_classes = {
        "train": Yolov5TrainArgs,
    }

    def capture_pre_run_state(self, config, config_path):
        super().capture_pre_run_state(config)

    @skip_inactive_stage
    def test_train_checkpoint_load(self, setup):
        model_file = os.path.join(self.configs["train"].project, "weights", "last.pt")
        assert os.path.isfile(model_file)
        val(model_file)

    @skip_inactive_stage
    def test_train_metrics(self, setup):
        results_file = os.path.join(self.configs["train"].project, "results.csv")
        assert os.path.isfile(results_file)
        metrics_df = pd.read_csv(results_file)
        assert metrics_df["epochs"][-1] == self.configs["train"].epochs
        metrics_key = "metrics" + "/" + self.targets["train"]["metric_name"]
        assert (
            self.targets["train"]["target"] - self.metrics["std"]
            <= metrics_df[metrics_key]
            <= self.targets["train"]["target"] + self.metrics["std"]
        )

    @skip_inactive_stage
    def test_export_onnx_graph(self, setup):
        onnx_model = onnx.load(
            os.path.join(self.configs["export"].weights, "last.onnx")
        )

        nodes = onnx_model.graph.node
        nodes_names = [node.name for node in nodes]
        nodes_count = Counter([node_name.split("_")[0] for node_name in nodes_names])
