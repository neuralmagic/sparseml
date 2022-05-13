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
import onnxruntime as ort
import pytest
import torch

from tests.integrations.base_tester import (
    BaseIntegrationManager,
    BaseIntegrationTester,
    skip_inactive_stage,
)
from tests.integrations.helpers import get_configs_with_cadence
from tests.integrations.object_detection.object_detection_args import (
    Yolov5ExportArgs,
    Yolov5TrainArgs,
)
from yolov5.export import load_checkpoint
from yolov5.val import run as val


try:
    import deepsparse
except Exception:
    deepsparse = None


METRIC_TO_INDEX = {"map0.5": 2}


class ObjectDetectionManager(BaseIntegrationManager):

    command_stubs = {
        "train": "sparseml.object_detection.train",
        "export": "sparseml.object_detection.export",
        "deploy": "sparseml.object_detection.deploy",
    }
    config_classes = {
        "train": Yolov5TrainArgs,
        "export": Yolov5ExportArgs,
    }

    def capture_pre_run_state(self):
        super().capture_pre_run_state()
        train_args = self.configs["train"].run_args
        export_args = self.configs["export"].run_args
        self.save_dir = tempfile.TemporaryDirectory(dir=train_args.project)
        train_args.project = self.save_dir.name

        if "export" in self.command_types:
            export_args.weights = os.path.join(
                train_args.project, "exp", "weights", "last.pt"
            )

    def teardown(self):
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
        manager = integration_manager
        model_file = os.path.join(manager.save_dir.name, "exp", "weights", "last.pt")
        assert os.path.isfile(model_file)
        model, extras = load_checkpoint(
            type_="val", weights=model_file, device=torch.device("cpu")
        )
        assert extras["ckpt"]["epoch"] == -1

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        manager = integration_manager
        train_args = manager.configs["train"]
        model_file = os.path.join(manager.save_dir.name, "exp", "weights", "last.pt")
        metrics, *_ = val(data=train_args.run_args.data, weights=model_file)
        if "target_name" in train_args.test_args:
            train_test_args = train_args.test_args
            metric_idx = METRIC_TO_INDEX[train_test_args["target_name"]]
            metric = metrics[metric_idx] * 100
            target_mean = train_test_args["target_mean"]
            target_std = train_test_args["target_std"]
            assert target_mean - target_std <= metric <= target_mean + target_std

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        manager = integration_manager
        onnx_file = os.path.join(
            os.path.dirname(manager.configs["export"]["args"].weights), "last.onnx"
        )
        assert os.path.isfile(onnx_file)
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)

    @pytest.mark.skipif(not deepsparse, reason="Deepsparse not installed")
    @skip_inactive_stage
    def test_export_target_model(self, integration_manager):
        manager = integration_manager
        export_args = manager.configs["export"]
        target_model_path = export_args.test_args.get("target_model")
        if not target_model_path:
            pytest.skip("No target model provided")
        run_model_path = os.path.join(
            os.path.dirname(export_args.run_args.weights), "last.onnx"
        )
        _, *_ = load_checkpoint(
            type_="val",
            weights=target_model_path,
            device=torch.device("cpu"),
        )
        input_data = deepsparse.utils.generate_random_inputs(run_model_path, 1)
        input_names = deepsparse.utils.get_input_names(run_model_path)
        output_names = deepsparse.utils.get_output_names(run_model_path)
        inputs_dict = {name: value for name, value in zip(input_names, input_data)}
        run_ort_sess = ort.InferenceSession(run_model_path)
        run_out = run_ort_sess.run(output_names, inputs_dict)
        target_ort_sess = ort.InferenceSession(target_model_path)
        target_out = target_ort_sess.run(output_names, inputs_dict)
        for ro, to in zip(run_out, target_out):
            pytest.approx(ro, abs=1e-5) == to
