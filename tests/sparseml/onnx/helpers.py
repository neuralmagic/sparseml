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
                    "onnx::Gemm_19": ([[16, 32, 1, 1]], None),
                    "onnx::Sigmoid_20": (None, None),
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
