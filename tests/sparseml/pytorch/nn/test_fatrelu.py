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

import os

import pytest
import torch

from sparseml.pytorch.nn import (
    FATReLU,
    convert_relus_to_fat,
    fat_exp_relu,
    fat_pw_relu,
    fat_relu,
    fat_sig_relu,
    set_relu_to_fat,
)
from tests.sparseml.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fat_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = 0.1
    out = fat_relu(x_tens, threshold, inplace=True)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = fat_relu(x_tens, threshold, inplace=False)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = FATReLU(threshold, inplace=True)(x_tens)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = FATReLU(threshold, inplace=False)(x_tens)
    assert (x_tens - out).sum() < 0.1


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fat_pw_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = torch.tensor(0.1)
    compression = torch.tensor(100.0)
    out = fat_pw_relu(x_tens, threshold, compression, inplace=True)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = fat_pw_relu(x_tens, threshold, compression, inplace=False)
    assert (x_tens - out).sum() < 0.1


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fat_sig_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = torch.tensor(0.1)
    compression = torch.tensor(100.0)
    out = fat_sig_relu(x_tens, threshold, compression)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = fat_sig_relu(x_tens, threshold, compression)
    assert (x_tens - out).sum() < 0.1


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fat_exp_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = torch.tensor(0.1)
    compression = torch.tensor(100.0)
    fat_exp_relu(x_tens, threshold, compression)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_convert_relus_to_fat():
    model = MLPNet()
    convert_relus_to_fat(model)

    for name, mod in model.named_modules():
        if "act" in name:
            assert isinstance(mod, FATReLU)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_set_relu_to_fat():
    model = MLPNet()
    layer_desc = MLPNet.layer_descs()[1]

    set_relu_to_fat(model, layer_desc.name)

    for name, mod in model.named_modules():
        if name == layer_desc.name:
            assert isinstance(mod, FATReLU)
