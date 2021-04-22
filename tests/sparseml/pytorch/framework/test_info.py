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
from sparseml.pytorch.framework import detect_framework, framework_info, is_supported


def test_is_supported():
    assert is_supported(Framework.pytorch)
    assert is_supported("pytorch")
    assert is_supported("/path/to/model.pth")
    assert is_supported("/path/to/model.pt")
    assert is_supported("/path/to/model.ptc")
    assert is_supported("/path/to/model.mar")

    from torch.nn import Sequential

    model = Sequential()
    assert is_supported(model)


def test_detect_framework():
    assert detect_framework(Framework.pytorch) == Framework.pytorch
    assert detect_framework("pytorch") == Framework.pytorch
    assert detect_framework("/path/to/model.pth") == Framework.pytorch
    assert detect_framework("/path/to/model.pt") == Framework.pytorch
    assert detect_framework("/path/to/model.ptc") == Framework.pytorch
    assert detect_framework("/path/to/model.mar") == Framework.pytorch

    from torch.nn import Sequential

    model = Sequential()
    assert detect_framework(model) == Framework.pytorch


def test_framework_info():
    base_info = base_framework_info(Framework.pytorch)
    info = framework_info()
    assert base_info == info

    assert info.framework == Framework.pytorch
    assert "torch" in info.package_versions
    assert "torchvision" in info.package_versions
    assert "onnx" in info.package_versions
    assert info.sparsification
    assert len(info.inference_providers) == 2
    assert info.training_available
    assert info.sparsification_available
    assert info.exporting_onnx_available
    assert info.inference_available
