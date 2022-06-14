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
import pytest
import torch
from pydantic import BaseModel

from sparseml.pytorch.models import ModelRegistry
from sparsezoo import Zoo
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
from tests.integrations.image_classification.args import (
    ImageClassificationExportArgs,
    ImageClassificationTrainArgs,
)


class ImageClassificationManager(BaseIntegrationManager):

    command_stubs = {
        "train": "sparseml.image_classification.train",
        "export": "sparseml.image_classification.export_onnx",
        "deploy": "sparseml.image_classification.deploy",  # placeholder
    }
    config_classes = {
        "train": ImageClassificationTrainArgs,
        "export": ImageClassificationExportArgs,
        "deploy": BaseModel,
    }

    def capture_pre_run_state(self):
        super().capture_pre_run_state()

        train_args = None
        self.save_dir = None

        if "train" in self.configs:
            train_args = self.configs["train"].run_args
            self.save_dir = tempfile.TemporaryDirectory()
            train_args.save_dir = self.save_dir.name
            train_args.logs_dir = os.path.join(self.save_dir.name, "tensorboard_logs")
            self.expected_checkpoint_path = os.path.join(
                train_args.save_dir, train_args.model_tag, "framework", "model.pth"
            )

        if "export" in self.configs:
            export_args = self.configs["export"].run_args
            if not self.save_dir:
                self.save_dir = tempfile.TemporaryDirectory()
                export_args.save_dir = self.save_dir.name
            else:
                export_args.checkpoint_path = self.expected_checkpoint_path
                export_args.save_dir = train_args.save_dir

    def teardown(self):
        """
        Cleanup environment after test completion
        """
        pass


class TestImageClassification(BaseIntegrationTester):
    @pytest.fixture(
        params=get_configs_with_cadence(
            os.environ.get("NM_TEST_CADENCE", "commit"), os.path.dirname(__file__)
        ),
        scope="class",
    )
    def integration_manager(self, request):
        manager = ImageClassificationManager(config_path=request.param)
        yield manager
        manager.teardown()

    @skip_inactive_stage
    def test_train_complete(self, integration_manager):
        # test checkpoint exists
        assert os.path.isfile(integration_manager.expected_checkpoint_path)
        # test that model can be reloaded from checkpoint
        # this includes reading arch_key and applying recipe before weight load
        create_kwargs = {}
        train_run_args = integration_manager.configs["train"].run_args
        if train_run_args.arch_key:
            create_kwargs["key"] = train_run_args.arch_key
        reloaded_model = ModelRegistry.create(
            pretrained_path=integration_manager.expected_checkpoint_path,
            **create_kwargs,
        )
        assert isinstance(reloaded_model, torch.nn.Module)

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        train_args = integration_manager.configs["train"]

        # locate training metrics file
        metrics_file_path = os.path.join(
            integration_manager.save_dir.name,
            train_args.run_args.model_tag,
            "model.txt",
        )
        assert os.path.isfile(metrics_file_path)

        if "target_name" not in train_args.test_args:
            pytest.skip("No target metric provided")

        # parse metrics from training
        metrics = {}
        with open(metrics_file_path) as metrics_file:
            for line in metrics_file.readlines():
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                metric, val = parts
                try:
                    val = float(val)
                except Exception:
                    continue  # cannot cast metric to float
                metrics[metric] = val

        assert train_args.test_args["target_name"] in metrics

        metric_val = metrics[train_args.test_args["target_name"]]
        expected_mean = train_args.test_args["target_mean"]
        expected_std = train_args.test_args["target_std"]
        assert (
            (expected_mean - expected_std)
            <= metric_val
            <= (expected_mean + expected_std)
        )

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        export_args = integration_manager.configs["export"]
        expected_onnx_path = os.path.join(
            integration_manager.save_dir.name,
            export_args.run_args.model_tag,
            "model.onnx",
        )
        assert os.path.isfile(expected_onnx_path)
        onnx_model = onnx.load(expected_onnx_path)
        onnx.checker.check_model(onnx_model)

    @skip_inactive_stage
    def test_export_target_model(self, integration_manager):
        # get exported and target model paths
        export_args = integration_manager.configs["export"]
        target_model_path = export_args.test_args.get("target_model")
        if not target_model_path:
            pytest.skip("No target model provided")
        if target_model_path.startswith("zoo:"):
            # download zoo model
            zoo_model = Zoo.load_model_from_stub(target_model_path)
            target_model_path = zoo_model.onnx_file.downloaded_path()
        export_model_path = os.path.join(
            integration_manager.save_dir.name,
            export_args.run_args.model_tag,
            "model.onnx",
        )

        model_op_counts_test(export_model_path, target_model_path)

        compare_outputs = export_args.test_args.get("compare_outputs", True)
        if isinstance(compare_outputs, str) and (
            compare_outputs.lower() in ["none", "False"]
        ):
            compare_outputs = False
        if compare_outputs:
            model_inputs_outputs_test(export_model_path, target_model_path)
