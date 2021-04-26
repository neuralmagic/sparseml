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

from sparseml.base import Framework
from sparseml.framework import framework_info as base_framework_info
from sparseml.keras.framework import detect_framework, framework_info, is_supported


def test_is_supported():
    assert is_supported(Framework.keras)
    assert is_supported("keras")
    assert is_supported("/path/to/model.h5")
    assert is_supported("/path/to/model.pb")

    from sparseml.keras.base import keras

    model = keras.Model()
    assert is_supported(model)


def test_detect_framework():
    assert detect_framework(Framework.keras) == Framework.keras
    assert detect_framework("keras") == Framework.keras
    assert detect_framework("/path/to/model.h5") == Framework.keras
    assert detect_framework("/path/to/model.pb") == Framework.keras

    from sparseml.keras.base import keras

    model = keras.Model()
    assert is_supported(model)


def test_framework_info():
    base_info = base_framework_info(Framework.keras)
    info = framework_info()
    assert base_info == info

    assert info.framework == Framework.keras
    assert "keras" in info.package_versions
    assert "keras2onnx" in info.package_versions
    assert "tf2onnx" in info.package_versions
    assert info.sparsification
    assert len(info.inference_providers) == 2
    assert info.training_available
    assert info.sparsification_available
    assert info.exporting_onnx_available
    assert info.inference_available
