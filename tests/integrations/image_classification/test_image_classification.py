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
from collections import defaultdict

import numpy
import onnx
import pytest
import torch
from onnxruntime import InferenceSession

from sparseml.onnx.utils import get_tensor_shape
from sparseml.pytorch.models import ModelRegistry
from sparsezoo import Zoo
from tests.integrations.base_tester import (
    BaseIntegrationManager,
    BaseIntegrationTester,
    skip_inactive_stage,
)
from tests.integrations.helpers import get_configs_with_cadence
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
    command_args_classes = {
        "train": ImageClassificationTrainArgs,
        "export": ImageClassificationExportArgs,
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
        reloaded_model = ModelRegistry.create(
            pretrained_path=integration_manager.expected_checkpoint_path
        )
        assert isinstance(reloaded_model, torch.nn.Module)

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        train_args = integration_manager.configs["train"]

        # locate training metrics file
        metrics_file_path = os.path.join(
            integration_manager.save_dir.name,
            train_args.run_args.model_tag,
            "framework",
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
            expected_mean - expected_std <= metric_val <= (expected_mean + expected_std)
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

        _test_model_op_counts(export_model_path, target_model_path)
        _test_model_inputs_outputs(export_model_path, target_model_path)


_TEST_OPS = {
    "MatMul",
    "Gemm",
    "Conv",
    "MatMulInteger",
    "ConvInteger",
    "QLinearMatMul",
    "QLinearConv",
}


def _test_model_op_counts(model_path_a, model_path_b):

    model_a = onnx.load(model_path_a)
    model_b = onnx.load(model_path_b)

    def _get_model_op_counts(model):
        op_counts = defaultdict(int)
        for node in model.graph.node:
            if node.op_type in _TEST_OPS:
                op_counts[node.op_type] += 1
        return op_counts

    op_counts_a = _get_model_op_counts(model_a)
    op_counts_b = _get_model_op_counts(model_b)

    assert len(op_counts_a) > 0
    assert len(op_counts_a) == len(op_counts_b)

    for op, count_a in op_counts_a.items():
        assert op in op_counts_b
        assert count_a == op_counts_b[op]


def _test_model_inputs_outputs(model_path_a, model_path_b):
    # compare export and target graphs and build fake data
    model_a = onnx.load(model_path_a)
    model_b = onnx.load(model_path_b)
    assert len(model_a.graph.input) == len(model_b.graph.input)
    assert len(model_a.graph.output) == len(model_b.graph.output)

    sample_input = {}
    output_names = []

    for input_a, input_b in zip(model_a.graph.input, model_b.graph.input):
        assert input_a.name == input_b.name
        input_a_shape = get_tensor_shape(input_a)
        assert input_a_shape == get_tensor_shape(input_b)
        sample_input[input_a.name] = numpy.random.randn(*input_a_shape).astype(
            numpy.float32
        )

    for output_a, output_b in zip(model_a.graph.output, model_b.graph.output):
        assert output_a.name == output_b.name
        assert get_tensor_shape(output_a) == get_tensor_shape(output_b)
        output_names.append(output_a.name)

    # run sample forward and test absolute max diff
    ort_sess_a = InferenceSession(model_path_a)
    ort_sess_b = InferenceSession(model_path_b)
    forward_output_a = ort_sess_a.run(output_names, sample_input)
    forward_output_b = ort_sess_b.run(output_names, sample_input)
    for out_a, out_b in zip(forward_output_a, forward_output_b):
        assert numpy.max(numpy.abs(out_a - out_b)) <= 1e-4
