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


import psutil

from sparseml.onnx.utils.model import ORTModelRunner, max_available_cores
from tests.sparseml.helpers import (  # noqa: F401
    OnnxModelDataFixture,
    model_test,
    onnx_models_with_data,
)


def test_max_available_cores():
    max_cores_available = max_available_cores()
    assert max_cores_available == psutil.cpu_count(logical=False)


def test_ort_model_runner(onnx_models_with_data: OnnxModelDataFixture):  # noqa: F811
    model_test(
        onnx_models_with_data.model_path,
        onnx_models_with_data.input_paths,
        onnx_models_with_data.output_paths,
        ORTModelRunner,
    )
