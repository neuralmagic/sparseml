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
import os

import pytest

from sparseml.onnx.optim import ModelAnalyzer, NodeAnalyzer
from sparsezoo import Zoo


from tests.sparseml.onnx.helpers import analyzer_models  # noqa isort: skip


GENERATE_TEST_FILES = os.getenv("NM_ML_GENERATE_ONNX_TEST_DATA", False)
GENERATE_TEST_FILES = False if GENERATE_TEST_FILES == "0" else GENERATE_TEST_FILES

RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(
    scope="session",
    params=[
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet_v1",
                "sub_architecture": "50",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "sparse_name": "base",
                "sparse_category": "none",
                "sparse_target": None,
            },
            "resnet50pytorch.json",
        ),
    ],
)
def analyzer_models_repo(request):
    model_args, output_path = request.param
    output_path = os.path.join(RELATIVE_PATH, "test_analyzer_model_data", output_path)
    model = Zoo.load_model(**model_args)
    model_path = model.onnx_file.downloaded_path()

    if GENERATE_TEST_FILES:
        analyzer = ModelAnalyzer(model_path)
        analyzer.save_json(output_path)

    output = {}
    with open(output_path) as output_file:
        output = dict(json.load(output_file))

    return model_path, output


def test_node_analyzer_kwargs():
    kwargs = {
        "id": "id",
        "op_type": "Conv",
        "input_names": ["in1", "in2"],
        "output_names": ["out1"],
        "input_shapes": [[16, 3, 3, 3]],
        "output_shapes": [[16, 16, 2, 2]],
        "flops": 27712,
        "params": 448,
        "prunable": True,
        "prunable_params": 432,
        "prunable_params_zeroed": 0,
        "weight_name": "conv.section.1.weight",
        "weight_shape": [16, 3, 3, 3],
        "bias_name": "conv.section.1.bias",
        "bias_shape": [16],
        "attributes": {"kernel": 1},
    }

    node = NodeAnalyzer(model=None, node=None, **kwargs)
    for key in kwargs:
        if key == "id":
            assert node.id_ == kwargs[key]
        else:
            assert getattr(node, key) == kwargs[key]

    assert node.prunable


def test_mode_analyzer_json():
    params = {
        "nodes": [
            {
                "id": "id",
                "op_type": "Conv",
                "input_names": ["in1", "in2"],
                "output_names": ["out1"],
                "input_shapes": [[16, 3, 3, 3]],
                "output_shapes": [[16, 16, 2, 2]],
                "flops": 27712,
                "params": 448,
                "prunable": True,
                "prunable_params_zeroed": 0,
                "weight_name": "conv.section.1.weight",
                "weight_shape": [16, 3, 3, 3],
                "bias_name": "conv.section.1.bias",
                "bias_shape": [16],
                "attributes": {"kernel": 1},
                "prunable_equation_sensitivity": None,
            },
            {
                "id": "id2",
                "op_type": "Gemm",
                "input_names": ["in1"],
                "output_names": ["out1"],
                "input_shapes": [[16, 32]],
                "output_shapes": [[16, 10]],
                "flops": 650,
                "params": 330,
                "prunable": True,
                "prunable_params_zeroed": 0,
                "weight_name": "conv.section.1.weight",
                "weight_shape": [10, 32],
                "bias_name": "conv.section.1.bias",
                "bias_shape": [10],
                "attributes": {"kernel": 1},
                "prunable_equation_sensitivity": None,
            },
        ]
    }
    nodes = [
        NodeAnalyzer(model=None, node=None, **params["nodes"][0]),
        NodeAnalyzer(model=None, node=None, **params["nodes"][1]),
    ]

    analyzer = ModelAnalyzer.from_dict(params)
    assert sorted(analyzer.nodes, key=lambda node: node.id_) == sorted(
        nodes, key=lambda node: node.id_
    )


def _test_model_analyzer(model_path: str, expected_output: str):
    analyzer = ModelAnalyzer(model_path)

    analyzer_from_json = ModelAnalyzer.from_dict(expected_output)
    print(analyzer.dict())
    assert analyzer.dict() == expected_output
    assert analyzer == analyzer_from_json


def test_model_analyzer(analyzer_models):  # noqa: F811
    model_path, expected_output = analyzer_models
    _test_model_analyzer(model_path, expected_output)


def test_model_analyzer_from_repo(analyzer_models_repo):
    model_path, expected_output = analyzer_models_repo
    _test_model_analyzer(model_path, expected_output)
