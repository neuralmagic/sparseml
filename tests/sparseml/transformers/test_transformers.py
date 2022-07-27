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

import glob
import math
import os
import shutil
from collections import Counter, OrderedDict

import onnx
import onnxruntime as ort
import pytest
from transformers import AutoConfig

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.transformers.sparsification import Trainer
from sparsezoo import Model
from sparsezoo.utils import load_numpy_list
from src.sparseml.transformers import export_transformer_to_onnx, load_task_model


def _is_yaml_recipe_present(model_path):
    return any(
        [
            file
            for file in glob.glob(os.path.join(model_path, "*"))
            if (
                file.endswith(
                    ".yaml",
                )
                or ("recipe" in file)
            )
        ]
    )


def _run_inference_onnx(path_onnx, input_data):
    ort_sess = ort.InferenceSession(path_onnx)
    model = onnx.load(path_onnx)
    input_names = [inp.name for inp in model.graph.input]

    model_input = OrderedDict(
        [(k, v.reshape(1, -1)) for k, v in zip(input_names, input_data.values())]
    )

    output = ort_sess.run(
        None,
        model_input,
    )
    return output


def _compare_onnx_models(model_1, model_2):
    major_nodes = [
        "QLinearMatMul",
        "Gemm",
        "MatMul",
        "MatMulInteger",
        "Conv",
        "QLinearConv",
        "ConvInteger",
        "QuantizeLinear",
        "DeQuantizeLinear",
    ]

    nodes1 = model_1.graph.node
    nodes1_names = [node.name for node in nodes1]
    nodes1_count = Counter([node_name.split("_")[0] for node_name in nodes1_names])

    nodes2 = model_2.graph.node
    nodes2_names = [node.name for node in nodes2]
    nodes2_count = Counter([node_name.split("_")[0] for node_name in nodes2_names])

    for node in major_nodes:
        assert nodes1_count[node] == nodes2_count[node]



@pytest.mark.parametrize(
    "model_stub, recipe_present, task",
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-conservative",  # noqa: E501
            False,
            "question-answering",
        ),
        (
            "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/12layer_pruned80_quant-none-vnni",  # noqa: E501
            False,
            "sentiment-analysis",
        ),
    ],
    scope="function",
)
class TestModelFromZoo:
    @pytest.fixture()
    def setup(self, model_stub, recipe_present, task):
        # setup
        self.onnx_retrieved_name = "retrieved_model.onnx"
        model = Model(model_stub)

        yield model, recipe_present, task

        # teardown
        model_path = model.path
        shutil.rmtree(model_path)

    def test_load_weights_apply_recipe(self, setup):
        model, recipe_present, task = setup
        model_path = model.training.default.path

        config = AutoConfig.from_pretrained(model_path)
        network = load_task_model(task, model_path, config)

        assert model
        assert recipe_present == _is_yaml_recipe_present(model_path)

        if recipe_present:
            trainer = Trainer(
                model=network,
                model_state_path=model_path,
                recipe=None,
                recipe_args=None,
                teacher=None,
            )
            applied = trainer.apply_manager(epoch=math.inf, checkpoint=None)

            assert applied

    def test_export_to_onnx(self, setup):
        model, recipe_present, task = setup
        path_onnx = model.onnx_model.path
        model_path = model.training.default.path

        path_retrieved_onnx = export_transformer_to_onnx(
            task=task,
            model_path=model_path,
            onnx_file_name=self.onnx_retrieved_name,
        )

        zoo_model = onnx.load(path_onnx)
        export_model = onnx.load(os.path.join(model_path, path_retrieved_onnx))

        assert export_model

        onnx.checker.check_model(export_model)
        _compare_onnx_models(zoo_model, export_model)

    def test_outputs_ort(self, setup):

        model, recipe_present, task = setup
        inputs_path = model.sample_inputs.path
        path_onnx = model.onnx_model.path
        model_path = model.training.default.path

        input_data = load_numpy_list(inputs_path)[0]

        path_retrieved_onnx = export_transformer_to_onnx(
            task=task,
            model_path=model_path,
            onnx_file_name=self.onnx_retrieved_name,
            sequence_length=next(iter(input_data.values())).shape[0],
        )

        out1 = _run_inference_onnx(path_onnx, input_data)
        out2 = _run_inference_onnx(path_retrieved_onnx, input_data)
        for o1, o2 in zip(out1, out2):
            pytest.approx(o1, abs=1e-5) == o2
