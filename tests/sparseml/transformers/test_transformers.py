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
import tarfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from transformers import AutoConfig

from sparseml.transformers.sparsification import Trainer
from sparsezoo import Zoo
from src.sparseml.transformers import export_transformer_to_onnx, load_task_model


def _is_yaml_recipe_present(model_path):
    return any(
        [
            file
            for file in glob.glob(os.path.join(model_path, "*"))
            if (file.endswith(".yaml") or ("recipe" in file))
        ]
    )


def _run_inference_onnx(path_onnx, input):
    ort_sess = ort.InferenceSession(path_onnx)
    with np.load(input) as data:
        input_0, input_1, input_2 = (
            data["input_0"].reshape(1, -1),
            data["input_1"].reshape(1, -1),
            data["input_2"].reshape(1, -1),
        )
    output = ort_sess.run(
        None,
        {"input_ids": input_0, "attention_mask": input_1, "token_type_ids": input_2},
    )
    return output


def _compare_onnx_models(model1, model2):
    optional_nodes_model1 = [
        "If",
        "Equal",
        "Gather",
        "Shape",
        # ops above are those which are used in the
        # original graph to create logits and softmax heads
        "Constant",
        "Cast",
    ]  # ops above are the remaining optional nodes
    optional_nodes_model2 = [
        "Constant",
        "Squeeze",
    ]  # ops above are
    # used in the original graph to create
    # logits and softmax heads

    nodes1 = model1.graph.node
    nodes1_names = [node.name for node in nodes1]

    nodes2 = model2.graph.node
    nodes2_names = [node.name for node in nodes2]

    # Extract ops which are in nodes1 but not in nodes2
    nodes1_names_diff = [
        node_name for node_name in nodes1_names if node_name not in nodes2_names
    ]

    # Extract ops which are in nodes2 but not in nodes1
    nodes2_names_diff = [
        node_name for node_name in nodes2_names if node_name not in nodes1_names
    ]
    # Assert that there are no important ops names in
    # nodes1_names_diff or nodes2_names_diff
    assert not [
        x for x in nodes1_names_diff if x.split("_")[0] not in optional_nodes_model1
    ]
    assert not [
        x for x in nodes2_names_diff if x.split("_")[0] not in optional_nodes_model2
    ]

    # Compare the structure of nodes which share names across m1 and m2
    for node1 in nodes1:
        if node1.name in set(nodes1_names).intersection(set(nodes2_names)):
            for node2 in nodes2:
                if node1.name == node2.name:
                    _compare_onnx_nodes(node1, node2)


def _compare_onnx_nodes(n1, n2):
    # checking for consistent lengths seems like a sufficient test for now.
    # due to internal structure, the naming of graph nodes
    # may vary, even thought the semantics remain unchanged.
    assert len(n1.input) == len(n2.input)
    assert len(n1.output) == len(n2.output)
    assert len(n1.op_type) == len(n2.op_type)
    assert len(n1.attribute) == len(n2.attribute)


@pytest.mark.parametrize(
    "model_stub, recipe_present, task",
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-conservative",  # noqa: E501
            False,
            "question-answering",
        )
    ],
    scope="function",
)
class TestModelFromZoo:
    @pytest.fixture()
    def setup(self, model_stub, recipe_present, task):
        # setup
        model = Zoo.load_model_from_stub(model_stub)
        model.download()

        path_onnx = model.onnx_file.downloaded_path()
        model_path = os.path.join(os.path.dirname(path_onnx), "pytorch")

        yield path_onnx, model_path, recipe_present, task

        # teardown
        shutil.rmtree(os.path.dirname(model_path))

    def test_load_weights_apply_recipe(self, setup):
        path_onnx, model_path, recipe_present, task = setup
        config = AutoConfig.from_pretrained(model_path)
        model = load_task_model(task, model_path, config)

        assert model
        assert recipe_present == _is_yaml_recipe_present(model_path)
        if recipe_present:

            trainer = Trainer(
                model=model,
                model_state_path=model_path,
                recipe=None,
                recipe_args=None,
                teacher=None,
            )
            applied = trainer.apply_manager(epoch=math.inf, checkpoint=None)

            assert applied

    def test_outputs(self, setup):
        path_onnx, model_path, recipe_present, task = setup
        path_retrieved_onnx = export_transformer_to_onnx(
            task=task,
            model_path=model_path,
            onnx_file_name="retrieved_model.onnx",
        )

        inputs_tar_path = os.path.join(
            os.path.dirname(path_onnx), "sample-inputs.tar.gz"
        )
        my_tar = tarfile.open(inputs_tar_path)
        my_tar.extractall(model_path)
        my_tar.close()

        inputs = glob.glob(os.path.join(model_path, "sample-inputs/*"))
        for input in inputs:
            out1 = _run_inference_onnx(path_onnx, input)
            out2 = _run_inference_onnx(path_retrieved_onnx, input)
            for o1, o2 in zip(out1, out2):
                pytest.approx(o1, abs=1e-5) == o2

    def test_export_to_onnx(self, setup):
        path_onnx, model_path, recipe_present, task = setup
        path_retrieved_onnx = export_transformer_to_onnx(
            task=task,
            model_path=model_path,
            onnx_file_name="retrieved_model.onnx",
        )

        m1 = onnx.load(path_onnx)
        m2 = onnx.load(os.path.join(model_path, path_retrieved_onnx))

        assert m2

        _compare_onnx_models(m1, m2)
