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
from sparseml.deepsparse.framework import detect_framework, framework_info, is_supported
from sparseml.framework import framework_info as base_framework_info
from sparsezoo import Model


def test_is_supported():
    assert is_supported(Framework.deepsparse)
    assert is_supported("deepsparse")
    assert is_supported("/path/to/model.onnx")

    model = Model(
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
    )
    assert is_supported(model)
    assert is_supported(model.onnx_model)


def test_detect_framework():
    assert detect_framework(Framework.deepsparse) == Framework.deepsparse
    assert detect_framework("deepsparse") == Framework.deepsparse
    assert detect_framework("/path/to/model.onnx") == Framework.deepsparse

    model = Model(
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
    )
    assert detect_framework(model) == Framework.deepsparse
    assert detect_framework(model.onnx_model) == Framework.deepsparse


def test_framework_info():
    base_info = base_framework_info(Framework.deepsparse)
    info = framework_info()
    assert base_info == info

    assert info.framework == Framework.deepsparse
    assert "deepsparse" in info.package_versions
    assert info.sparsification
    assert len(info.inference_providers) == 1
    assert not info.training_available
    assert not info.sparsification_available
    assert not info.exporting_onnx_available
    assert info.inference_available
