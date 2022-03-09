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

import tempfile

import pytest

from sparseml.base import Framework
from sparseml.framework import (
    FrameworkInferenceProviderInfo,
    FrameworkInfo,
    framework_info,
    load_framework_info,
    save_framework_info,
)
from sparseml.sparsification import SparsificationInfo


@pytest.mark.parametrize(
    "const_args",
    [
        {
            "name": "test name",
            "description": "test description",
            "device": "test device",
        },
        {
            "name": "test name",
            "description": "test description",
            "device": "test device",
            "supported_sparsification": SparsificationInfo(),
            "available": True,
            "properties": {"prop_key": "prop_val"},
            "warnings": ["test warning"],
        },
    ],
)
def test_framework_inference_provider_info_lifecycle(const_args):
    # test construction
    info = FrameworkInferenceProviderInfo(**const_args)
    assert info, "No object returned for info constructor"

    # test serialization
    info_str = info.json()
    assert info_str, "No json returned for info"

    # test deserialization
    info_reconst = FrameworkInferenceProviderInfo.parse_raw(info_str)
    assert info == info_reconst, "Reconstructed does not equal original"


@pytest.mark.parametrize(
    "const_args",
    [
        {
            "framework": Framework.unknown,
            "package_versions": {"test": "0.1.0"},
        },
        {
            "framework": Framework.unknown,
            "package_versions": {"test": "0.1.0"},
            "sparsification": SparsificationInfo(),
            "inference_providers": [
                FrameworkInferenceProviderInfo(
                    name="test", description="test", device="test"
                )
            ],
            "properties": {"test_prop": "val"},
            "training_available": True,
            "sparsification_available": True,
            "exporting_onnx_available": True,
            "inference_available": True,
        },
    ],
)
def test_framework_info_lifecycle(const_args):
    # test construction
    info = FrameworkInfo(**const_args)
    assert info, "No object returned for info constructor"

    # test serialization
    info_str = info.json()
    assert info_str, "No json returned for info"

    # test deserialization
    info_reconst = FrameworkInfo.parse_raw(info_str)
    assert info == info_reconst, "Reconstructed does not equal original"


def test_framework_info():
    # test that unknown raises an exception,
    # other frameworks will test in their packages
    with pytest.raises(ValueError):
        framework_info(Framework.unknown)


def test_save_load_framework_info():
    info = FrameworkInfo(
        framework=Framework.unknown, package_versions={"unknown": "0.0.1"}
    )
    save_framework_info(info)
    loaded_json = load_framework_info(info.json())
    assert info == loaded_json

    test_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    save_framework_info(info, test_path)
    loaded_path = load_framework_info(test_path)
    assert info == loaded_path
