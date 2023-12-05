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

from sparsezoo import Model


__all__ = [
    "onnx_repo_models",
    "GENERATE_TEST_FILES",
]

TEMP_FOLDER = os.path.expanduser(os.path.join("~", ".cache", "nm_models"))
RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))
GENERATE_TEST_FILES = os.getenv("NM_ML_GENERATE_ONNX_TEST_DATA", False)
GENERATE_TEST_FILES = False if GENERATE_TEST_FILES == "0" else GENERATE_TEST_FILES


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
        input_paths = model.sample_inputs.path
    if model.sample_outputs is not None:
        output_paths = model.sample_outputs["framework"].path

    return OnnxRepoModelFixture(model_path, model_name, input_paths, output_paths)
