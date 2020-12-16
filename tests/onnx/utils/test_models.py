import os
from typing import Any, Callable, Dict, List, NamedTuple

import psutil
import pytest
from neuralmagicML.onnx.utils.data import DataLoader
from neuralmagicML.onnx.utils.model import (
    ModelRunner,
    NMAnalyzeModelRunner,
    NMModelRunner,
    ORTModelRunner,
    max_available_cores,
)
from neuralmagicML.utils import RepoModel
from onnx import load_model

try:
    import neuralmagic
except ModuleNotFoundError:
    neuralmagic = None


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
                "architecture": "resnet-v1",
                "sub_architecture": "50",
                "dataset": "imagenet",
                "framework": "pytorch",
                "desc": "base",
            }
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet-v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "framework": "pytorch",
                "desc": "base",
            }
        ),
    ]
)
def onnx_models_with_data(request) -> OnnxModelDataFixture:
    model_args = request.param
    model = RepoModel(**model_args)
    model_path = model.download_onnx_file(overwrite=False)
    data_paths = model.download_data_files(overwrite=False)
    inputs_paths = None
    outputs_paths = None
    for path in data_paths:
        if "_sample-inputs" in path:
            inputs_paths = path.split(".tar")[0]
        elif "_sample-outputs" in path:
            outputs_paths = path.split(".tar")[0]
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

    input_glob = os.path.join(input_paths, "*")
    output_glob = os.path.join(output_paths, "*")

    dataloader = DataLoader(input_glob, output_glob, 2, 0)
    model_runner = runner_constructor(model, batch_size=2)
    outputs, _ = model_runner.run(dataloader)
    _test_output(outputs, dataloader, batch_size=2)

    dataloader = DataLoader(input_glob, output_glob, 1, 0)
    model_runner = runner_constructor(model, batch_size=1,)

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
    neuralmagic is None, reason="neuralmagic is not installed on the system"
)
def test_nm_model_runner(onnx_models_with_data: OnnxModelDataFixture):
    _test_model(
        onnx_models_with_data.model_path,
        onnx_models_with_data.input_paths,
        onnx_models_with_data.output_paths,
        NMModelRunner,
    )


@pytest.mark.skipif(
    neuralmagic is None, reason="neuralmagic is not installed on the system"
)
def test_nm_analyze_model_runner(onnx_models_with_data: OnnxModelDataFixture,):
    model = load_model(onnx_models_with_data.model_path)

    # Sanity check, asserting model can run random input
    dataloader = DataLoader.from_model_random(model, 5, 0, 10)
    model_runner = NMAnalyzeModelRunner(model, batch_size=5)
    outputs, _ = model_runner.run(dataloader, max_steps=5)
    fields = ["num_threads", "num_sockets", "average_total_time", "iteration_times"]
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
