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

import json
import math
import os
import tempfile

import onnx
import onnxruntime as ort
import pytest

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.transformers.utils import SparseAutoModel
from tests.integrations.base_tester import (
    BaseIntegrationManager,
    BaseIntegrationTester,
    skip_inactive_stage,
)
from tests.integrations.helpers import get_configs_with_cadence
from tests.integrations.transformers.transformers_args import (
    MaskedLanguageModellingArgs,
    QuestionAnsweringArgs,
    TextClassificationArgs,
    TokenClassificationArgs,
    TransformersTrainArgs,
)


try:
    import deepsparse
except Exception:
    deepsparse = None


class TransformersManager(BaseIntegrationManager):
    command_stubs = {
        "train": "sparseml.transformers.train.{task}",
        "export": "sparseml.transformers.export",
    }
    config_classes = {"train": TransformersTrainArgs}
    task_config_classes = {
        "masked_language_modeling": MaskedLanguageModellingArgs,
        "question_answering": QuestionAnsweringArgs,
        "text_classification": TextClassificationArgs,
        "token_classification": TokenClassificationArgs,
    }
    supported_metrics = ["f1", "exact_match"]

    def capture_pre_run_state(self):
        super().capture_pre_run_state()
        train_args = (
            self.configs["train"].run_args if "train" in self.command_types else None
        )
        export_args = (
            self.configs["export"].run_args if "export" in self.command_types else None
        )
        self.save_dir = tempfile.TemporaryDirectory(
            dir=os.path.dirname(train_args.output_dir)
        )
        if train_args:
            train_args.output_dir = self.save_dir.name

    def get_root_commands(self, raw_configs):
        self.task = (
            raw_configs["train"]["task"].lower().replace("-", "_")
            if "train" in self.command_types
            else "NullTask"
        )
        if self.task not in self.task_config_classes:
            raise ValueError(f"{self.task} is not a supported task")

        self.config_classes["train"] = self.task_config_classes[self.task]

        command_stubs_final = self.command_stubs
        command_stubs_final["train"] = command_stubs_final["train"].format(
            task=self.task
        )
        return command_stubs_final


class TestTransformers(BaseIntegrationTester):
    @pytest.fixture(
        params=get_configs_with_cadence(
            os.environ.get("NM_TEST_CADENCE", "commit"), os.path.dirname(__file__)
        ),
        scope="class",
    )
    def integration_manager(self, request):
        manager = TransformersManager(config_path=request.param)
        yield manager
        manager.teardown()

    @skip_inactive_stage
    def test_train_complete(self, integration_manager):
        manager = integration_manager
        run_args = manager.configs["train"].run_args
        results_file = os.path.join(manager.save_dir.name, "train_results.json")
        model_file = os.path.join(manager.save_dir.name, "pytorch_model.bin")
        assert os.path.isfile(model_file)
        end_epoch = (
            ScheduledModifierManager.from_yaml(run_args.recipe).max_epochs
            if run_args.recipe
            else run_args.num_train_epochs
        )
        with open(results_file) as f:
            train_results = json.load(f)
        assert train_results["epoch"] == math.floor(end_epoch)

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        manager = integration_manager
        args = manager.configs["train"]
        results_file = os.path.join(manager.save_dir.name, "eval_results.json")
        with open(results_file) as f:
            eval_results = json.load(f)
        if "target_name" in args.test_args:
            train_test_args = args.test_args
            if train_test_args["target_name"] not in manager.supported_metrics:
                raise ValueError(
                    f"{train_test_args['target_name']} is not a supported target metric"
                )
            metric = eval_results["eval_f1"]
            target_mean = train_test_args["target_mean"]
            target_std = train_test_args["target_std"]
            assert target_mean - target_std <= metric <= target_mean + target_std

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        manager = integration_manager
        export_run_args = manager.configs["export"].run_args
        onnx_file = os.path.join(
            os.path.dirname(export_run_args.model_path), export_run_args.onnx_file_name
        )
        assert os.path.isfile(onnx_file)
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)

    @skip_inactive_stage
    def test_export_target_model(self, integration_manager):
        manager = integration_manager
        export_args = manager.configs["export"]
        target_model_path = export_args.test_args.get("target_model")
        if not target_model_path:
            pytest.skip("No target model provided")
        run_model_path = os.path.join(manager.save_dir.name, "pytorch_model.bin")
        model = _load_model_on_task(run_model_path, "student", manager.task)

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


def _load_model_on_task(model_name_or_path, model_type, task, **model_kwargs):
    load_funcs = {
        "masked_language_modeling": SparseAutoModel.masked_language_modeling_from_pretrained,  # noqa
        "question_answering": SparseAutoModel.question_answering_from_pretrained,
        "text_classification": SparseAutoModel.text_classification_from_pretrained,
        "token_classification": SparseAutoModel.token_classification_from_pretrained,
    }
    return load_funcs[task](model_name_or_path, model_type=model_type, **model_kwargs)
