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

import psutil
import pytest
from onnx import load_model

from sparseml.onnx.utils.data import DataLoader
from sparseml.onnx.utils.model import (
    DeepSparseAnalyzeModelRunner,
    DeepSparseModelRunner,
    ModelRunner,
    ORTModelRunner,
    max_available_cores,
)
from sparsezoo import Model


try:
    import deepsparse
except ModuleNotFoundError:
    deepsparse = None


OnnxModelDataFixture = NamedTuple(
    "OnnxModelDataFixture",
    [("model_path", str), ("input_paths", str), ("output_paths", str)],
)


@pytest.fixture(
    params=[
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",
    ]
)
def onnx_models_with_data(request) -> OnnxModelDataFixture:
    model_stub = request.param
    model = Model(model_stub)
    model_path = model.onnx_model.path
    inputs_paths = None
    outputs_paths = None
    if model.sample_inputs is not None:
        inputs_paths = model.sample_inputs.path
    if model.sample_outputs is not None:
        outputs_paths = model.sample_outputs["framework"].path
    return OnnxModelDataFixture(model_path, inputs_paths, outputs_paths)


def test_max_available_cores():
    max_cores_available = max_available_cores()
    assert max_cores_available == psutil.cpu_count(logical=False)


def _test_output(outputs: Dict[str, List], dataloader: DataLoader, batch_size: int = 1):
    _, reference_output = dataloader.labeled_data[0]
    for out in outputs:
        for out_key, reference_key in zip(out, reference_output):
            reference_shape = reference_output[reference_key].shape
            assert out[out_key].shape == (batch_size,) + reference_shape
            assert out[out_key].dtype == reference_output[reference_key].dtype


def _test_model(
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


def test_ort_model_runner(onnx_models_with_data: OnnxModelDataFixture):
    _test_model(
        onnx_models_with_data.model_path,
        onnx_models_with_data.input_paths,
        onnx_models_with_data.output_paths,
        ORTModelRunner,
    )


@pytest.mark.skipif(
    deepsparse is None, reason="deepsparse is not installed on the system"
)
def test_nm_model_runner(onnx_models_with_data: OnnxModelDataFixture):
    _test_model(
        onnx_models_with_data.model_path,
        onnx_models_with_data.input_paths,
        onnx_models_with_data.output_paths,
        DeepSparseModelRunner,
    )


@pytest.mark.skipif(
    deepsparse is None, reason="deepsparse is not installed on the system"
)
def test_nm_analyze_model_runner(
    onnx_models_with_data: OnnxModelDataFixture,
):
    model = load_model(onnx_models_with_data.model_path)

    # Sanity check, asserting model can run random input
    dataloader = DataLoader.from_model_random(model, 5, 0, 10)
    model_runner = DeepSparseAnalyzeModelRunner(model, batch_size=5)
    outputs, _ = model_runner.run(dataloader, max_steps=5)
    fields = ["num_threads", "average_total_time", "iteration_times"]
    layer_fields = [
        "name",
        "canonical_name",
        "input_dims",
        "output_dims",
        "strides",
        "required_flops",
        "kernel_sparsity",
        "activation_sparsity",
        "average_run_time_in_ms",
        "average_utilization",
        "average_teraflops_per_second",
    ]
    for out in outputs:
        for field in fields:
            assert field in out
        for layer_info in out["layer_info"]:
            for field in layer_fields:
                assert field in layer_info
