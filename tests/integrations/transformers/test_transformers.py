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

import collections
import inspect
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import onnx
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

from flaky import flaky
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.transformers.export import load_task_model
from sparseml.transformers.utils import SparseAutoModel
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
from tests.integrations.transformers.args import (
    MaskedLanguageModellingArgs,
    QuestionAnsweringArgs,
    TextClassificationArgs,
    TokenClassificationArgs,
    TransformersDeployArgs,
    TransformersExportArgs,
)


deepsparse_error = None
try:
    from deepsparse import Pipeline
except Exception as e:
    deepsparse_error = e


class TransformersManager(BaseIntegrationManager):
    command_stubs = {
        "train": "sparseml.transformers.train.{task}",
        "export": "sparseml.transformers.export_onnx",
        "deploy": None,
    }
    config_classes = {
        "train": None,
        "export": TransformersExportArgs,
        "deploy": TransformersDeployArgs,
    }
    task_config_classes = {
        "masked_language_modeling": MaskedLanguageModellingArgs,
        "question_answering": QuestionAnsweringArgs,
        "text_classification": TextClassificationArgs,
        "token_classification": TokenClassificationArgs,
    }
    supported_metrics = ["f1", "exact_match"]

    def capture_pre_run_state(self):
        super().capture_pre_run_state()
        self._check_deploy_requirements(deepsparse_error)

        # Setup temporary directory for train run
        if "train" in self.configs:
            train_args = self.configs["train"].run_args
            os.makedirs(train_args.output_dir, exist_ok=True)
            self.save_dir = tempfile.TemporaryDirectory(dir=train_args.output_dir)
            train_args.output_dir = self.save_dir.name

    def save_stage_information(self, command_type):
        # Either grab output directory from train run or setup new temporary directory
        # for export
        if command_type == "export":
            export_args = self.configs["export"].run_args
            if not self.save_dir:
                self.save_dir = tempfile.TemporaryDirectory()
                export_args.save_dir = self.save_dir.name
            else:
                train_args = self.configs["train"].run_args
                checkpoints = [
                    file
                    for file in os.listdir(train_args.output_dir)
                    if os.path.isdir(os.path.join(train_args.output_dir, file))
                    and file.startswith("checkpoint-")
                ]
                checkpoints.sort(key=lambda ckpt: ckpt.split("-")[1])
                export_args.model_path = (
                    os.path.join(train_args.output_dir, checkpoints[-1])
                    if checkpoints
                    else train_args.output_dir
                )
            self.commands["export"] = self.configs["export"].create_command_script()

        # Grab onnx output path from the export stage if it exists
        if command_type == "deploy":
            deploy_args = self.configs["deploy"].run_args
            if self.save_dir:
                export_args = self.configs["export"].run_args
                deploy_args.model_path = export_args.model_path

    def add_abridged_configs(self):
        if "train" in self.command_types:
            self.configs["train"].run_args.max_train_samples = 2
            self.configs["train"].run_args.max_eval_samples = 2

    def get_root_commands(self, raw_configs):
        self.task = (
            raw_configs["train"]["task"].lower().replace("-", "_")
            if "train" in self.command_types
            else "NullTask"
        )
        if self.task not in self.task_config_classes:
            raise ValueError(f"{self.task} is not a supported task")

        self.config_classes["train"] = self.task_config_classes[self.task]

        command_stubs_final = deepcopy(self.command_stubs)
        command_stubs_final["train"] = command_stubs_final["train"].format(
            task=self.task
        )
        return command_stubs_final

    def teardown(self):
        pass  # not yet implemented


@flaky(max_runs=2, min_passes=1)
class TestTransformers(BaseIntegrationTester):
    @pytest.fixture(
        params=get_configs_with_cadence(
            os.environ.get("SPARSEML_TEST_CADENCE", "pre-commit"),
            os.path.dirname(__file__),
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
        model_directory = manager.save_dir.name
        assert os.path.isdir(model_directory)
        assert os.path.exists(os.path.join(model_directory, "pytorch_model.bin"))
        model = _load_model_on_task(model_directory, "student", manager.task)
        assert isinstance(model, torch.nn.Module)

        end_epoch = (
            ScheduledModifierManager.from_yaml(run_args.recipe).max_epochs
            if run_args.recipe
            else run_args.num_train_epochs
        )
        # skip for step-based tests
        if end_epoch:
            with open(results_file) as f:
                train_results = json.load(f)
            assert abs(train_results["epoch"] - math.floor(end_epoch)) < 0.1

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        manager = integration_manager
        args = manager.configs["train"]
        if args.run_args.one_shot:
            pytest.skip("One-shot mode. Skipping test")
        results_file = os.path.join(manager.save_dir.name, "eval_results.json")
        with open(results_file) as f:
            eval_results = json.load(f)
        if "target_name" in args.test_args:
            train_test_args = args.test_args
            if train_test_args["target_name"] not in manager.supported_metrics:
                raise ValueError(
                    f"{train_test_args['target_name']} is not a supported target metric"
                )
            metric = eval_results[train_test_args["target_name"]]
            target_mean = train_test_args["target_mean"]
            target_std = train_test_args["target_std"]
            assert (target_mean - target_std) <= metric <= (target_mean + target_std)

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        manager = integration_manager
        onnx_file = _get_onnx_model_path(manager)
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
        export_model_path = _get_onnx_model_path(manager)
        model_op_counts_test(export_model_path, target_model_path)

        compare_outputs = export_args.test_args.get("compare_outputs", True)
        if isinstance(compare_outputs, str) and (
            compare_outputs.lower() in ["none", "False"]
        ):
            compare_outputs = False
        if compare_outputs:
            model_inputs_outputs_test(
                export_model_path,
                target_model_path,
                _create_bert_input,
                model_path=export_args.run_args.model_path,
                task=manager.task,
            )

    @skip_inactive_stage
    def test_deploy_model_compile(self, integration_manager):
        manager = integration_manager
        args = manager.configs["deploy"]
        _ = Pipeline.create(
            task=args.run_args.task,
            model_path=os.path.dirname(_get_onnx_model_path(manager)),
        )


def _get_onnx_model_path(manager) -> str:
    export_run_args = manager.configs["export"].run_args
    return os.path.join(
        Path(export_run_args.model_path).parents[0],
        "deployment",
        export_run_args.onnx_file_name,
    )


def _load_model_on_task(model_name_or_path, model_type, task, **model_kwargs):
    load_funcs = {
        "masked_language_modeling": SparseAutoModel.masked_language_modeling_from_pretrained,  # noqa
        "question_answering": SparseAutoModel.question_answering_from_pretrained,
        "text_classification": SparseAutoModel.text_classification_from_pretrained,
        "token_classification": SparseAutoModel.token_classification_from_pretrained,
    }
    return load_funcs[task](model_name_or_path, model_type=model_type, **model_kwargs)


def _create_bert_input(model_path, task):
    task = task.replace("_", "-")
    config_args = {"finetuning_task": task} if task else {}
    config = AutoConfig.from_pretrained(
        model_path,
        **config_args,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=384)

    model = load_task_model(task, model_path, config)

    # create fake model input
    inputs = tokenizer(
        "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
    ).data  # Dict[Tensor]

    # Rearrange inputs' keys to match those defined by model foward func, which
    # seem to define how the order of inputs is determined in the exported model
    forward_args_spec = inspect.getfullargspec(model.__class__.forward)
    dropped = [f for f in inputs.keys() if f not in forward_args_spec.args]
    inputs = collections.OrderedDict(
        [
            (f, inputs[f][0].reshape(1, -1))
            for f in forward_args_spec.args
            if f in inputs
        ]
    )
    if dropped:
        raise ValueError(
            "The following inputs were not present in the model forward function "
            f"and therefore dropped from ONNX export: {dropped}"
        )

    input_names = list(inputs.keys())
    inputs = tuple([inputs[f] for f in input_names])

    return inputs
