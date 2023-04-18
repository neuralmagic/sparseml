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
from pathlib import Path

import onnx
import pandas as pd
import pytest
import torch

from flaky import flaky
from tests.integrations.base_tester import (
    BaseIntegrationManager,
    BaseIntegrationTester,
    skip_inactive_stage,
)
from tests.integrations.helpers import (
    get_configs_with_cadence,
    model_inputs_outputs_test,
    model_op_counts_test,
)
from tests.integrations.yolov5.args import (
    Yolov5DeployArgs,
    Yolov5ExportArgs,
    Yolov5TrainArgs,
)
from yolov5.models.common import DetectMultiBackend


METRIC_TO_COLUMN = {"map0.5": "metrics/mAP_0.5"}

deepsparse_error = None
try:
    from deepsparse import Pipeline
except Exception as e:
    deepsparse_error = e


class Yolov5Manager(BaseIntegrationManager):

    command_stubs = {
        "train": "sparseml.yolov5.train",
        "export": "sparseml.yolov5.export_onnx",
        "deploy": None,
    }
    config_classes = {
        "train": Yolov5TrainArgs,
        "export": Yolov5ExportArgs,
        "deploy": Yolov5DeployArgs,
    }

    def capture_pre_run_state(self):
        super().capture_pre_run_state()
        self._check_deploy_requirements(deepsparse_error)

        self.save_dir = None
        # Setup temporary directory for train run
        if "train" in self.configs:
            train_args = self.configs["train"].run_args
            directory = os.path.dirname(train_args.project)
            os.makedirs(directory, exist_ok=True)
            self.save_dir = tempfile.TemporaryDirectory(dir=directory)
            train_args.project = self.save_dir.name
            self.expected_checkpoint_path = os.path.join(
                train_args.project,
                "exp",
                "weights",
                "last.pt",
            )

        # Either grab output directory from train run or setup new temporary directory
        # for export
        if "export" in self.configs:
            export_args = self.configs["export"].run_args
            if self.save_dir:
                export_args.weights = self.expected_checkpoint_path

        if "deploy" in self.configs:
            deploy_args = self.configs["deploy"].run_args
            export_args = self.configs["export"].run_args
            deploy_args.model_path = _get_onnx_file_path(
                export_args.weights, export_args.one_shot
            )

        # Turn on "_" -> "-" conversion for CLI args
        for stage, config in self.configs.items():
            config.dashed_keywords = True

    def add_abridged_configs(self):
        pass  # max train and eval steps phased out

    def teardown(self):
        if "train" in self.command_types:
            self.save_dir.cleanup()


@flaky(max_runs=2, min_passes=1)
class TestYolov5(BaseIntegrationTester):
    @pytest.fixture(
        params=get_configs_with_cadence(
            os.environ.get("SPARSEML_TEST_CADENCE", "pre-commit"),
            os.path.dirname(__file__),
        ),
        scope="class",
    )
    def integration_manager(self, request):
        manager = Yolov5Manager(config_path=request.param)
        yield manager
        manager.teardown()

    @skip_inactive_stage
    def test_train_complete(self, integration_manager):
        # Test that file is created
        manager = integration_manager
        model_file = manager.expected_checkpoint_path
        assert os.path.isfile(model_file)

        # Test that model file loadable
        model = DetectMultiBackend(model_file, device=torch.device("cpu"))
        assert model.model.sparsified

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
        export_args = manager.configs["export"].run_args
        onnx_file = _get_onnx_file_path(export_args.weights, export_args.one_shot)

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

        export_model_path = _get_onnx_file_path(
            export_args.run_args.weights, export_args.run_args.one_shot
        )

        # Downloads model if zoo stubs and additionally tests that it can be loaded
        _ = DetectMultiBackend(target_model_path, device=torch.device("cpu"))

        model_op_counts_test(export_model_path, target_model_path)

        compare_outputs = export_args.test_args.get("compare_outputs", True)
        if isinstance(compare_outputs, str) and (
            compare_outputs.lower() in ["none", "False"]
        ):
            compare_outputs = False
        if compare_outputs:
            model_inputs_outputs_test(export_model_path, target_model_path)

    @skip_inactive_stage
    def test_deploy_model_compile(self, integration_manager):
        manager = integration_manager
        args = manager.configs["deploy"]
        _ = Pipeline.create("yolo", model_path=args.run_args.model_path)


def _get_onnx_file_path(weights, one_shot):
    weights = Path(weights)

    if str(weights).startswith("zoo:") or not len(weights.parents):
        sub_dir = (
            str(weights).split("zoo:")[1].replace("/", "_")
            if str(weights).startswith("zoo:")
            else weights
        )

        if one_shot:
            one_shot_str = str(weights).split("zoo:")[1].replace("/", "_")
            sub_dir = f"{sub_dir}_one_shot_{one_shot_str}"

        save_dir = Path(os.getcwd()) / "DeepSparse_Deployment" / sub_dir
        onnx_file_name = "model.onnx"

    else:
        save_dir = (
            weights.parents[1] / "DeepSparse_Deployment"
            if weights.parent.stem == "weights"
            else weights.parent / "DeepSparse_Deployment"
        )
        onnx_file_name = str(Path(weights.name).with_suffix(".onnx"))

    return str(save_dir / onnx_file_name)
