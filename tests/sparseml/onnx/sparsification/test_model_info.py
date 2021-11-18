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

import pytest

from sparseml.onnx.sparsification import ModelInfo
from tests.sparseml.onnx.utils.test_helpers import get_prunable_onnx_model


@pytest.mark.parametrize(
    "model,expected_dict",
    [
        (
            get_prunable_onnx_model(),
            {
                "metadata": {},
                "analysis_results": [],
                "layer_info": {
                    "node1.weight": {
                        "name": "node1.weight",
                        "op_type": "conv",
                        "params": 54,
                        "prunable": True,
                        "execution_order": 0,
                        "attributes": {
                            "in_channels": 3,
                            "out_channels": 2,
                            "kernel_shape": [3, 3],
                            "groups": 1,
                            "stride": [1, 1],
                            "padding": [0, 0, 0, 0],
                            "sparsity": 0.0,
                            "node_name": "node1",
                            "node_output_id": "B",
                            "first_prunable_layer": True,
                        },
                        "bias_params": 2,
                        "flops": None,
                    },
                    "node2.weight": {
                        "name": "node2.weight",
                        "op_type": "linear",
                        "params": 36,
                        "prunable": True,
                        "execution_order": 1,
                        "attributes": {
                            "in_channels": 12,
                            "out_channels": 3,
                            "sparsity": 0.0,
                            "node_name": "node2",
                            "node_output_id": "C",
                        },
                        "bias_params": None,
                        "flops": None,
                    },
                    "node3.weight": {
                        "name": "node3.weight",
                        "op_type": "linear",
                        "params": 12,
                        "prunable": True,
                        "execution_order": 2,
                        "attributes": {
                            "in_channels": 3,
                            "out_channels": 4,
                            "sparsity": 0.0,
                            "node_name": "node3",
                            "node_output_id": "D",
                            "last_prunable_layer": True,
                        },
                        "bias_params": None,
                        "flops": None,
                    },
                },
            },
        ),
    ],
)
def test_onnx_model_info(model, expected_dict):
    model_info = ModelInfo(model)
    assert model_info.to_dict() == expected_dict
