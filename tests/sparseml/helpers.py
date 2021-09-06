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

from typing import Any, Callable, Dict, List, NamedTuple

import pytest
from onnx import load_model

from sparseml.onnx.utils.data import DataLoader
from sparseml.onnx.utils.model import ModelRunner
from sparsezoo import Zoo


__all__ = [
    "model_test",
    "OnnxModelDataFixture",
    "onnx_models_with_data",
]


OnnxModelDataFixture = NamedTuple(
    "OnnxModelDataFixture",
    [("model_path", str), ("input_paths", str), ("output_paths", str)],
)


@pytest.fixture(
    params=[
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "sparse_name": "base",
                "sparse_category": "none",
                "sparse_target": None,
            }
        ),
    ]
)
def onnx_models_with_data(request) -> OnnxModelDataFixture:
    model_args = request.param
    model = Zoo.load_model(**model_args)
    model_path = model.onnx_file.downloaded_path()
    data_paths = [data_file.downloaded_path() for data_file in model.data.values()]
    inputs_paths = None
    outputs_paths = None
    for path in data_paths:
        if "sample-inputs" in path:
            inputs_paths = path
        elif "sample-outputs" in path:
            outputs_paths = path
    return OnnxModelDataFixture(model_path, inputs_paths, outputs_paths)


def _test_output(outputs: Dict[str, List], dataloader: DataLoader, batch_size: int = 1):
    _, reference_output = dataloader.labeled_data[0]
    for out in outputs:
        for out_key, reference_key in zip(out, reference_output):
            reference_shape = reference_output[reference_key].shape
            assert out[out_key].shape == (batch_size,) + reference_shape
            assert out[out_key].dtype == reference_output[reference_key].dtype


def model_test(
    model_path: str,
    input_paths: str,
    output_paths: str,
    runner_constructor: Callable[[Any], ModelRunner],
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    model = load_model(model_path)

    dataloader = DataLoader(input_paths, output_paths, 2, 0)
    model_runner = runner_constructor(model, batch_size=2)
    outputs, _ = model_runner.run(dataloader)
    _test_output(outputs, dataloader, batch_size=2)

    dataloader = DataLoader(input_paths, output_paths, 1, 0)
    model_runner = runner_constructor(
        model,
        batch_size=1,
    )

    outputs, _ = model_runner.run(dataloader, max_steps=1)
    assert len(outputs) == 1

    outputs, _ = model_runner.run(dataloader)
    _test_output(outputs, dataloader)
