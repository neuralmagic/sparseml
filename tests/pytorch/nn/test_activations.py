import pytest

import sys
import os
import torch
import torch.nn.functional as TF
from torch.nn import ReLU as TReLU
from torch.nn import ReLU6 as TReLU6
from torch.nn import PReLU, LeakyReLU

from neuralmagicML.pytorch.nn import (
    ReLU,
    ReLU6,
    swish,
    Swish,
    replace_activation,
    create_activation,
    is_activation,
)

from tests.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_relu():
    x_tens = torch.randn(16, 1, 64, 64)
    comp_one = ReLU(num_channels=1)(x_tens)
    comp_two = TReLU()(x_tens)

    assert (comp_one - comp_two).abs().sum() < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_relu6():
    x_tens = torch.randn(16, 1, 64, 64)
    comp_one = ReLU6(num_channels=1)(x_tens)
    comp_two = TReLU6()(x_tens)

    assert (comp_one - comp_two).abs().sum() < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_swish():
    x_tens = torch.randn(16, 1, 64, 64)
    comp_one = swish(x_tens)
    comp_two = Swish(1)(x_tens)
    comp_three = x_tens * TF.sigmoid(x_tens)

    assert (comp_one - comp_two).abs().sum() < sys.float_info.epsilon
    assert (comp_one - comp_three).abs().sum() < sys.float_info.epsilon


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_replace_activation():
    model = MLPNet()
    layer_desc = MLPNet.layer_descs()[1]
    act = replace_activation(model, layer_desc.name, "relu6")

    assert isinstance(act, ReLU6)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_is_activation():
    assert is_activation(ReLU())
    assert is_activation(ReLU6())
    assert is_activation(TReLU())
    assert is_activation(TReLU6())
    assert is_activation(PReLU())
    assert is_activation(LeakyReLU())
    assert is_activation(Swish())
