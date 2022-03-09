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
import sys

import pytest
import torch
import torch.nn.functional as TF
from torch.nn import LeakyReLU, PReLU
from torch.nn import ReLU as TReLU
from torch.nn import ReLU6 as TReLU6

from sparseml.pytorch.nn import (
    ReLU,
    ReLU6,
    Swish,
    create_activation,
    is_activation,
    replace_activation,
    swish,
)
from tests.sparseml.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_relu():
    x_tens = torch.randn(16, 1, 64, 64)
    comp_one = ReLU(num_channels=1)(x_tens)
    comp_two = TReLU()(x_tens)

    assert (comp_one - comp_two).abs().sum() < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_relu6():
    x_tens = torch.randn(16, 1, 64, 64)
    comp_one = ReLU6(num_channels=1)(x_tens)
    comp_two = TReLU6()(x_tens)

    assert (comp_one - comp_two).abs().sum() < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_swish():
    x_tens = torch.randn(16, 1, 64, 64)
    comp_one = swish(x_tens)
    comp_two = Swish(1)(x_tens)
    comp_three = x_tens * TF.sigmoid(x_tens)

    assert (comp_one - comp_two).abs().sum() < sys.float_info.epsilon
    assert (comp_one - comp_three).abs().sum() < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_replace_activation():
    model = MLPNet()
    layer_desc = MLPNet.layer_descs()[1]
    act = replace_activation(model, layer_desc.name, "relu6")

    assert isinstance(act, ReLU6)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_create_activation():
    act = create_activation("relu", inplace=True, num_channels=1)
    assert isinstance(act, ReLU)

    act = create_activation("relu6", inplace=True, num_channels=1)
    assert isinstance(act, ReLU6)

    act = create_activation("prelu", inplace=True, num_channels=1)
    assert isinstance(act, PReLU)

    act = create_activation("lrelu", inplace=True, num_channels=1)
    assert isinstance(act, LeakyReLU)

    act = create_activation("swish", inplace=True, num_channels=1)
    assert isinstance(act, Swish)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_is_activation():
    assert is_activation(ReLU())
    assert is_activation(ReLU6())
    assert is_activation(TReLU())
    assert is_activation(TReLU6())
    assert is_activation(PReLU())
    assert is_activation(LeakyReLU())
    assert is_activation(Swish())
