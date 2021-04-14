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
from sparseml.tensorflow_v1.base import tf_compat
from sparseml.tensorflow_v1.framework import (
    detect_framework,
    framework_info,
    is_supported,
)


def test_is_supported():
    assert is_supported(Framework.tensorflow_v1)
    assert is_supported("tensorflow_v1")
    assert is_supported("/path/to/model.pb")

    with tf_compat.Graph().as_default() as graph:
        assert is_supported(graph)
        with tf_compat.Session() as session:
            assert is_supported(session)


def test_detect_framework():
    assert detect_framework(Framework.tensorflow_v1) == Framework.tensorflow_v1
    assert detect_framework("tensorflow_v1") == Framework.tensorflow_v1
    assert detect_framework("/path/to/model.pb") == Framework.tensorflow_v1

    with tf_compat.Graph().as_default() as graph:
        assert detect_framework(graph) == Framework.tensorflow_v1
        with tf_compat.Session() as session:
            assert detect_framework(session) == Framework.tensorflow_v1


def test_framework_info():
    base_info = base_framework_info(Framework.tensorflow_v1)
    info = framework_info()
    assert base_info == info

    assert info.framework == Framework.tensorflow_v1
    assert "tensorflow" in info.package_versions
    assert "tf2onnx" in info.package_versions
    assert "onnx" in info.package_versions
    assert info.sparsification
    assert len(info.inference_providers) == 2
    assert info.training_available
    assert info.sparsification_available
    assert info.exporting_onnx_available
    assert info.inference_available
