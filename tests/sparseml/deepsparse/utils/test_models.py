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
from onnx import load_model

from sparseml.deepsparse.utils.model import (
    DeepSparseAnalyzeModelRunner,
    DeepSparseModelRunner,
)
from sparseml.onnx.utils.data import DataLoader
from tests.sparseml.helpers import (  # noqa: F401
    OnnxModelDataFixture,
    model_test,
    onnx_models_with_data,
)


try:
    import deepsparse
except ModuleNotFoundError:
    deepsparse = None


@pytest.mark.skipif(
    deepsparse is None, reason="deepsparse is not installed on the system"
)
def test_nm_model_runner(onnx_models_with_data: OnnxModelDataFixture):  # noqa: F811
    model_test(
        onnx_models_with_data.model_path,
        onnx_models_with_data.input_paths,
        onnx_models_with_data.output_paths,
        DeepSparseModelRunner,
    )


@pytest.mark.skipif(
    deepsparse is None, reason="deepsparse is not installed on the system"
)
def test_nm_analyze_model_runner(
    onnx_models_with_data: OnnxModelDataFixture,  # noqa: F811
):
    model = load_model(onnx_models_with_data.model_path)

    # Sanity check, asserting model can run random input
    dataloader = DataLoader.from_model_random(model, 5, 0, 10)
    model_runner = DeepSparseAnalyzeModelRunner(model, batch_size=5)
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
