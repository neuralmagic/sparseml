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

import onnx
import pandas as pd
import pytest
import torch

from tests.integrations.base_tester import (
    BaseIntegrationManager,
    BaseIntegrationTester,
    skip_inactive_stage,
)
from tests.integrations.helpers import (
    get_configs_with_cadence,
    test_model_inputs_outputs,
    test_model_op_counts,
)
from tests.integrations.object_detection.args import Yolov5ExportArgs, Yolov5TrainArgs
from yolov5.export import load_checkpoint


METRIC_TO_COLUMN = {"map0.5": "metrics/mAP_0.5"}


class ObjectDetectionManager(BaseIntegrationManager):

    command_stubs = {
        "train": "sparseml.object_detection.train",
        "export": "sparseml.object_detection.export_onnx",
    }
    config_classes = {
        "train": Yolov5TrainArgs,
        "export": Yolov5ExportArgs,
    }

    def capture_pre_run_state(self):
        super().capture_pre_run_state()

        # Setup temporary directory for train run
        if "train" in self.command_types:
            train_args = self.configs["train"].run_args
            directory = os.path.dirname(train_args.project)
            os.makedirs(directory, exist_ok=True)
            self.save_dir = tempfile.TemporaryDirectory(dir=directory)
            train_args.project = self.save_dir.name

        # Either grab output directory from train run or setup new temporary directory
        # for export
        if "export" in self.command_types:
            export_args = self.configs["export"].run_args
            export_args.weights = os.path.join(
                train_args.project,
                "exp",
                "weights",
                "last.pt" if "train" in self.command_types else export_args.weights,
            )

        # Turn on "_" -> "-" conversion for CLI args
        for stage, config in self.configs.items():
            config.dashed_keywords = True

    def teardown(self):
        if "train" in self.command_types:
            self.save_dir.cleanup()


class TestObjectDetection(BaseIntegrationTester):
    @pytest.fixture(
        params=get_configs_with_cadence(
            os.environ.get("NM_TEST_CADENCE", "commit"), os.path.dirname(__file__)
        ),
        scope="class",
    )
    def integration_manager(self, request):
        manager = ObjectDetectionManager(config_path=request.param)
        yield manager
        manager.teardown()

    @skip_inactive_stage
    def test_train_complete(self, integration_manager):
        # Test that file is created
        manager = integration_manager
        model_file = os.path.join(manager.save_dir.name, "exp", "weights", "last.pt")
        assert os.path.isfile(model_file)

        # Test that model file loadable
        _, extras = load_checkpoint(
            type_="val", weights=model_file, device=torch.device("cpu")
        )

        # Test that training ran to completion
        assert extras["ckpt"]["epoch"] == -1

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        # Test train metric(s) if specified
        manager = integration_manager
        train_args = manager.configs["train"]
        results_file = os.path.join(manager.save_dir.name, "exp", "results.csv")

        if "target_name" in train_args.test_args:
            train_test_args = train_args.test_args
            results = pd.read_csv(results_file, skipinitialspace=True)

            metric_key = METRIC_TO_COLUMN[train_test_args["target_name"]]
            metric = results[metric_key].iloc[-1] * 100

            target_mean = train_test_args["target_mean"]
            target_std = train_test_args["target_std"]

            assert target_mean - target_std <= metric <= target_mean + target_std

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        # Test that onnx model is loadable and passes onnx checker
        manager = integration_manager
        export_args = manager.configs["export"]
        onnx_file = os.path.join(
            os.path.dirname(export_args.run_args.weights), "last.onnx"
        )

        assert os.path.isfile(onnx_file)

        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)

    @skip_inactive_stage
    def test_export_target_model(self, integration_manager):
        # If target model provided in target_args, test that they have similar
        # ort output
        manager = integration_manager
        export_args = manager.configs["export"]
        target_model_path = export_args.test_args.get("target_model")

        if not target_model_path:
            pytest.skip("No target model provided")

        export_model_path = os.path.join(
            os.path.dirname(export_args.run_args.weights), "last.onnx"
        )

        # Downloads model if zoo stubs and additionally tests that it can be loaded
        _, *_ = load_checkpoint(
            type_="val", weights=target_model_path, device=torch.device("cpu")
        )

        test_model_op_counts(export_model_path, target_model_path)

        compare_outputs = export_args.test_args.get("compare_outputs", True)
        if isinstance(compare_outputs, str) and (
            compare_outputs.lower() in ["none", "False"]
        ):
            compare_outputs = False
        if compare_outputs:
            test_model_inputs_outputs(export_model_path, target_model_path)
