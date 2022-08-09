# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,t
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import NamedTuple

import pytest
import torch

from sparseml.pytorch.utils import ModuleExporter
from sparsezoo import Model
from tests.sparseml.pytorch.helpers import ConvNet, LinearNet, MLPNet


__all__ = [
    "extract_node_models",
    "analyzer_models",
    "onnx_repo_models",
    "GENERATE_TEST_FILES",
]

TEMP_FOLDER = os.path.expanduser(os.path.join("~", ".cache", "nm_models"))
RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))
GENERATE_TEST_FILES = os.getenv("NM_ML_GENERATE_ONNX_TEST_DATA", False)
GENERATE_TEST_FILES = False if GENERATE_TEST_FILES == "0" else GENERATE_TEST_FILES


@pytest.fixture(
    params=[
        [
            (
                "test_linear_net",
                LinearNet,
                torch.randn(8),
                {
                    "output": ([[8]], [[8]]),
                    "onnx::Add_10": ([[8]], [[16]]),
                    "onnx::MatMul_11": ([[16]], [[16]]),
                    "onnx::Add_13": ([[16]], [[32]]),
                    "onnx::MatMul_14": ([[32]], [[32]]),
                    "onnx::Add_16": ([[32]], [[16]]),
                    "onnx::MatMul_17": ([[16]], [[16]]),
                    "onnx::Add_19": ([[16]], [[8]]),
                    "input": (None, [[8]]),
                },
            ),
            None,
        ],
        [
            (
                "test_mlp_net",
                MLPNet,
                torch.randn(8),
                {
                    "output": ([[64]], [[64]]),
                    "onnx::Add_8": ([[8]], [[16]]),
                    "input.1": ([[16]], [[16]]),
                    "onnx::MatMul_10": ([[16]], [[16]]),
                    "onnx::Add_12": ([[16]], [[32]]),
                    "input.3": ([[32]], [[32]]),
                    "onnx::MatMul_14": ([[32]], [[32]]),
                    "onnx::Add_16": ([[32]], [[64]]),
                    "onnx::Sigmoid_17": ([[64]], [[64]]),
                    "input": (None, [[8]]),
                },
            ),
            None,
        ],
        [
            (
                "test_conv_net",
                ConvNet,
                torch.randn(16, 3, 3, 3),
                {
                    "output": (None, [[16, 10]]),
                    "input.1": ([[16, 3, 3, 3]], [[16, 16, 2, 2]]),
                    "input.4": ([[16, 16, 2, 2]], [[16, 16, 2, 2]]),
                    "input.8": ([[16, 16, 2, 2]], [[16, 32, 1, 1]]),
                    "input.12": ([[16, 32, 1, 1]], [[16, 32, 1, 1]]),
                    "onnx::Reshape_11": ([[16, 32, 1, 1]], [[16, 32, 1, 1]]),
                    "onnx::Gemm_17": ([[16, 32, 1, 1]], None),
                    "onnx::Sigmoid_18": (None, None),
                    "input": (None, [[16, 3, 3, 3]]),
                },
            ),
            None,
        ],
    ]
)
def extract_node_models(request):
    # we assume having two tests
    # - one for old version of PyTorch
    # - one for new version of PyTorch (1.10.2)
    params_python_legacy, params_python_upgrade = request.param

    # check if the test for new PyTorch version test is not `None`
    if params_python_upgrade:
        *_, expected_output_upgrade = params_python_upgrade
    else:
        expected_output_upgrade = None

    (
        model_name,
        model_function,
        sample_batch,
        expected_output_legacy,
    ) = params_python_legacy
    directory = os.path.join(TEMP_FOLDER, model_name)
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, "model.onnx")

    if not os.path.exists(model_path):
        module = model_function()
        exporter = ModuleExporter(module, directory)
        exporter.export_onnx(sample_batch=sample_batch)
    return (
        os.path.expanduser(model_path),
        expected_output_legacy,
        expected_output_upgrade,
    )


# TODO update when flops are done
# add a list for exact output
# [python10_output, python9_output]
# check whether each of those pass, if at least one passes than good to go.
@pytest.fixture(
    params=[
        [
            (
                "test_linear_net_batched",
                LinearNet,
                torch.randn(42, 8),
                {
                    "nodes": [
                        {
                            "id": "onnx::Gemm_9",
                            "op_type": "Gemm",
                            "input_names": ["input"],
                            "output_names": ["onnx::Gemm_9"],
                            "input_shapes": [[42, 8]],
                            "output_shapes": [[42, 16]],
                            "params": 144,
                            "prunable": True,
                            "prunable_params": 128,
                            "prunable_params_zeroed": 0,
                            "prunable_equation_sensitivity": 14.0,
                            "flops": 272.0,
                            "weight_name": "seq.fc1.weight",
                            "weight_shape": [16, 8],
                            "bias_name": "seq.fc1.bias",
                            "bias_shape": [16],
                            "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1},
                        },
                        {
                            "id": "onnx::Gemm_10",
                            "op_type": "Gemm",
                            "input_names": ["onnx::Gemm_9"],
                            "output_names": ["onnx::Gemm_10"],
                            "input_shapes": [[42, 16]],
                            "output_shapes": [[42, 32]],
                            "params": 544,
                            "prunable": True,
                            "prunable_params": 512,
                            "prunable_params_zeroed": 0,
                            "prunable_equation_sensitivity": 7.411764705882353,
                            "flops": 1056.0,
                            "weight_name": "seq.fc2.weight",
                            "weight_shape": [32, 16],
                            "bias_name": "seq.fc2.bias",
                            "bias_shape": [32],
                            "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1},
                        },
                        {
                            "id": "onnx::Gemm_11",
                            "op_type": "Gemm",
                            "input_names": ["onnx::Gemm_10"],
                            "output_names": ["onnx::Gemm_11"],
                            "input_shapes": [[42, 32]],
                            "output_shapes": [[42, 16]],
                            "params": 528,
                            "prunable": True,
                            "prunable_params": 512,
                            "prunable_params_zeroed": 0,
                            "prunable_equation_sensitivity": 7.636363636363637,
                            "flops": 1040.0,
                            "weight_name": "seq.block1.fc1.weight",
                            "weight_shape": [16, 32],
                            "bias_name": "seq.block1.fc1.bias",
                            "bias_shape": [16],
                            "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1},
                        },
                        {
                            "id": "output",
                            "op_type": "Gemm",
                            "input_names": ["onnx::Gemm_11"],
                            "output_names": ["output"],
                            "input_shapes": [[42, 16]],
                            "output_shapes": [[42, 8]],
                            "params": 136,
                            "prunable": True,
                            "prunable_params": 128,
                            "prunable_params_zeroed": 0,
                            "prunable_equation_sensitivity": 14.823529411764707,
                            "flops": 264.0,
                            "weight_name": "seq.block1.fc2.weight",
                            "weight_shape": [8, 16],
                            "bias_name": "seq.block1.fc2.bias",
                            "bias_shape": [8],
                            "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1},
                        },
                    ]
                },
            ),
            None,
        ],
        [
            (
                "test_linear_net",
                LinearNet,
                torch.randn(8),
                {
                    "nodes": [
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 256.0,
                            "id": "onnx::Add_10",
                            "input_names": ["input"],
                            "input_shapes": [[8]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_10"],
                            "output_shapes": [[16]],
                            "params": 128,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.375,
                            "prunable_params": 128,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_21",
                            "weight_shape": [8, 16],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 16.0,
                            "id": "onnx::MatMul_11",
                            "input_names": ["onnx::Add_10"],
                            "input_shapes": [[16]],
                            "op_type": "Add",
                            "output_names": ["onnx::MatMul_11"],
                            "output_shapes": [[16]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 1024.0,
                            "id": "onnx::Add_13",
                            "input_names": ["onnx::MatMul_11"],
                            "input_shapes": [[16]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_13"],
                            "output_shapes": [[32]],
                            "params": 512,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.1875,
                            "prunable_params": 512,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_22",
                            "weight_shape": [16, 32],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 32.0,
                            "id": "onnx::MatMul_14",
                            "input_names": ["onnx::Add_13"],
                            "input_shapes": [[32]],
                            "op_type": "Add",
                            "output_names": ["onnx::MatMul_14"],
                            "output_shapes": [[32]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 1024.0,
                            "id": "onnx::Add_16",
                            "input_names": ["onnx::MatMul_14"],
                            "input_shapes": [[32]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_16"],
                            "output_shapes": [[16]],
                            "params": 512,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.1875,
                            "prunable_params": 512,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_23",
                            "weight_shape": [32, 16],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 16.0,
                            "id": "onnx::MatMul_17",
                            "input_names": ["onnx::Add_16"],
                            "input_shapes": [[16]],
                            "op_type": "Add",
                            "output_names": ["onnx::MatMul_17"],
                            "output_shapes": [[16]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 256.0,
                            "id": "onnx::Add_19",
                            "input_names": ["onnx::MatMul_17"],
                            "input_shapes": [[16]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_19"],
                            "output_shapes": [[8]],
                            "params": 128,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.375,
                            "prunable_params": 128,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_24",
                            "weight_shape": [16, 8],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 8.0,
                            "id": "output",
                            "input_names": ["onnx::Add_19"],
                            "input_shapes": [[8]],
                            "op_type": "Add",
                            "output_names": ["output"],
                            "output_shapes": [[8]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                },
            ),
            None,
        ],
        [
            (
                "test_mlp_net",
                MLPNet,
                torch.randn(8),
                {
                    "nodes": [
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 256.0,
                            "id": "onnx::Add_8",
                            "input_names": ["input"],
                            "input_shapes": [[8]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_8"],
                            "output_shapes": [[16]],
                            "params": 128,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.375,
                            "prunable_params": 128,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_19",
                            "weight_shape": [8, 16],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 16.0,
                            "id": "input.1",
                            "input_names": ["onnx::Add_8"],
                            "input_shapes": [[16]],
                            "op_type": "Add",
                            "output_names": ["input.1"],
                            "output_shapes": [[16]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 16.0,
                            "id": "onnx::MatMul_10",
                            "input_names": ["input.1"],
                            "input_shapes": [[16]],
                            "op_type": "Relu",
                            "output_names": ["onnx::MatMul_10"],
                            "output_shapes": [[16]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 1024.0,
                            "id": "onnx::Add_12",
                            "input_names": ["onnx::MatMul_10"],
                            "input_shapes": [[16]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_12"],
                            "output_shapes": [[32]],
                            "params": 512,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.1875,
                            "prunable_params": 512,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_20",
                            "weight_shape": [16, 32],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 32.0,
                            "id": "input.3",
                            "input_names": ["onnx::Add_12"],
                            "input_shapes": [[32]],
                            "op_type": "Add",
                            "output_names": ["input.3"],
                            "output_shapes": [[32]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 32.0,
                            "id": "onnx::MatMul_14",
                            "input_names": ["input.3"],
                            "input_shapes": [[32]],
                            "op_type": "Relu",
                            "output_names": ["onnx::MatMul_14"],
                            "output_shapes": [[32]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 4096.0,
                            "id": "onnx::Add_16",
                            "input_names": ["onnx::MatMul_14"],
                            "input_shapes": [[32]],
                            "op_type": "MatMul",
                            "output_names": ["onnx::Add_16"],
                            "output_shapes": [[64]],
                            "params": 2048,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.09375,
                            "prunable_params": 2048,
                            "prunable_params_zeroed": 0,
                            "weight_name": "onnx::MatMul_21",
                            "weight_shape": [32, 64],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 64.0,
                            "id": "onnx::Sigmoid_17",
                            "input_names": ["onnx::Add_16"],
                            "input_shapes": [[64]],
                            "op_type": "Add",
                            "output_names": ["onnx::Sigmoid_17"],
                            "output_shapes": [[64]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 64.0,
                            "id": "output",
                            "input_names": ["onnx::Sigmoid_17"],
                            "input_shapes": [[64]],
                            "op_type": "Sigmoid",
                            "output_names": ["output"],
                            "output_shapes": [[64]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                },
            ),
            None,
        ],
        [
            (
                "test_conv_net",
                ConvNet,
                torch.randn(16, 3, 3, 3),
                {
                    "nodes": [
                        {
                            "attributes": {
                                "dilations": [1, 1],
                                "group": 1,
                                "kernel_shape": [3, 3],
                                "pads": [1, 1, 1, 1],
                                "strides": [2, 2],
                            },
                            "bias_name": "seq.conv1.bias",
                            "bias_shape": [16],
                            "flops": 27712.0,
                            "id": "input.1",
                            "input_names": ["input"],
                            "input_shapes": [[16, 3, 3, 3]],
                            "op_type": "Conv",
                            "output_names": ["input.1"],
                            "output_shapes": [[16, 16, 2, 2]],
                            "params": 448,
                            "prunable": True,
                            "prunable_equation_sensitivity": 7.703703703703703,
                            "prunable_params": 432,
                            "prunable_params_zeroed": 0,
                            "weight_name": "seq.conv1.weight",
                            "weight_shape": [16, 3, 3, 3],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 1024.0,
                            "id": "input.4",
                            "input_names": ["input.1"],
                            "input_shapes": [[16, 16, 2, 2]],
                            "op_type": "Relu",
                            "output_names": ["input.4"],
                            "output_shapes": [[16, 16, 2, 2]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {
                                "dilations": [1, 1],
                                "group": 1,
                                "kernel_shape": [3, 3],
                                "pads": [1, 1, 1, 1],
                                "strides": [2, 2],
                            },
                            "bias_name": "seq.conv2.bias",
                            "bias_shape": [32],
                            "flops": 73760.0,
                            "id": "input.8",
                            "input_names": ["input.4"],
                            "input_shapes": [[16, 16, 2, 2]],
                            "op_type": "Conv",
                            "output_names": ["input.8"],
                            "output_shapes": [[16, 32, 1, 1]],
                            "params": 4640,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.6620689655172414,
                            "prunable_params": 4608,
                            "prunable_params_zeroed": 0,
                            "weight_name": "seq.conv2.weight",
                            "weight_shape": [32, 16, 3, 3],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 512.0,
                            "id": "input.12",
                            "input_names": ["input.8"],
                            "input_shapes": [[16, 32, 1, 1]],
                            "op_type": "Relu",
                            "output_names": ["input.12"],
                            "output_shapes": [[16, 32, 1, 1]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 512.0,
                            "id": "onnx::Reshape_11",
                            "input_names": ["input.12"],
                            "input_shapes": [[16, 32, 1, 1]],
                            "op_type": "GlobalAveragePool",
                            "output_names": ["onnx::Reshape_11"],
                            "output_shapes": [[16, 32, 1, 1]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": None,
                            "id": "onnx::Gemm_17",
                            "input_names": ["onnx::Reshape_11"],
                            "input_shapes": [[16, 32, 1, 1]],
                            "op_type": "Reshape",
                            "output_names": ["onnx::Gemm_17"],
                            "output_shapes": None,
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                        {
                            "attributes": {"alpha": 1.0, "beta": 1.0, "transB": 1},
                            "bias_name": "mlp.fc.bias",
                            "bias_shape": [10],
                            "flops": 650.0,
                            "id": "onnx::Sigmoid_18",
                            "input_names": ["onnx::Gemm_17"],
                            "input_shapes": None,
                            "op_type": "Gemm",
                            "output_names": ["onnx::Sigmoid_18"],
                            "output_shapes": None,
                            "params": 330,
                            "prunable": True,
                            "prunable_equation_sensitivity": 0.0,
                            "prunable_params": 320,
                            "prunable_params_zeroed": 0,
                            "weight_name": "mlp.fc.weight",
                            "weight_shape": [10, 32],
                        },
                        {
                            "attributes": {},
                            "bias_name": None,
                            "bias_shape": None,
                            "flops": 160.0,
                            "id": "output",
                            "input_names": ["onnx::Sigmoid_18"],
                            "input_shapes": None,
                            "op_type": "Sigmoid",
                            "output_names": ["output"],
                            "output_shapes": [[16, 10]],
                            "params": 0,
                            "prunable": False,
                            "prunable_equation_sensitivity": None,
                            "prunable_params": -1,
                            "prunable_params_zeroed": 0,
                            "weight_name": None,
                            "weight_shape": None,
                        },
                    ]
                },
            ),
            None,
        ],
    ]
)
def analyzer_models(request):
    data_legacy_python, data_upgrade_python = request.param
    if data_upgrade_python:
        *_, expected_output_upgrade = data_upgrade_python
    else:
        expected_output_upgrade = None

    (
        model_name,
        model_function,
        sample_batch,
        expected_output_legacy,
    ) = data_legacy_python
    directory = os.path.join(TEMP_FOLDER, model_name)
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, "model.onnx")

    if not os.path.exists(model_path):
        module = model_function()
        exporter = ModuleExporter(module, directory)
        exporter.export_onnx(sample_batch=sample_batch)
    return (
        os.path.expanduser(model_path),
        expected_output_legacy,
        expected_output_upgrade,
    )


OnnxRepoModelFixture = NamedTuple(
    "OnnxRepoModelFixture",
    [
        ("model_path", str),
        ("model_name", str),
        ("input_paths", str),
        ("output_paths", str),
    ],
)


@pytest.fixture(
    scope="session",
    params=[
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
            "resnet50",
        ),
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",  # noqa 501
            "mobilenet",
        ),
    ],
)
def onnx_repo_models(request) -> OnnxRepoModelFixture:

    model_stub, model_name = request.param
    model = Model(model_stub)
    model_path = model.onnx_model.path
    input_paths, output_paths = None, None
    if model.sample_inputs is not None:
        if not model.sample_inputs.files:
            model.sample_inputs.unzip()
        input_paths = model.sample_inputs.path
    if model.sample_outputs is not None:
        if not model.sample_outputs["framework"].files:
            model.sample_outputs["framework"].unzip()
        output_paths = model.sample_outputs["framework"].path

    return OnnxRepoModelFixture(model_path, model_name, input_paths, output_paths)
