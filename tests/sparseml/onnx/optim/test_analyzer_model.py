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
from sparsezoo import Model
from tests.sparseml.onnx.helpers import GENERATE_TEST_FILES


RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))


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


@pytest.mark.parametrize(
    "name,stub",
    [
        (
            "resnet50-pq",
            "zoo:cv/classification/resnet_v1-50/"
            "pytorch/sparseml/imagenet/pruned95_quant-none",
        ),
        (
            "mobilenet-p",
            "zoo:cv/classification/mobilenet_v1-1.0"
            "/pytorch/sparseml/imagenet/pruned-moderate",
        ),
    ],
)
def test_model_analyzer(name, stub):
    model = Model(stub)
    analyzer = ModelAnalyzer(model.onnx_model.path)
    actual_analysis = analyzer.dict()

    data_path = os.path.join(RELATIVE_PATH, "test_analyzer_model_data", name + ".json")
    if GENERATE_TEST_FILES:
        with open(data_path, "w") as fp:
            json.dump(actual_analysis, fp, indent=1)

    with open(data_path) as fp:
        expected_analysis = json.load(fp)

    assert actual_analysis == expected_analysis
