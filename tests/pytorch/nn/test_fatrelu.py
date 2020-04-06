import torch

from neuralmagicML.pytorch.nn import (
    fat_relu,
    fat_pw_relu,
    fat_sig_relu,
    fat_exp_relu,
    FATReLU,
    convert_relus_to_fat,
    set_relu_to_fat,
)

from tests.pytorch.helpers import MLPNet


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


def test_fat_pw_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = torch.tensor(0.1)
    compression = torch.tensor(100.0)
    out = fat_pw_relu(x_tens, threshold, compression, inplace=True)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = fat_pw_relu(x_tens, threshold, compression, inplace=False)
    assert (x_tens - out).sum() < 0.1


def test_fat_sig_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = torch.tensor(0.1)
    compression = torch.tensor(100.0)
    out = fat_sig_relu(x_tens, threshold, compression)
    assert (x_tens - out).sum() < 0.1

    x_tens = torch.randn(1, 8, 64, 64)
    out = fat_sig_relu(x_tens, threshold, compression)
    assert (x_tens - out).sum() < 0.1


def test_fat_exp_relu():
    x_tens = torch.randn(1, 8, 64, 64)
    threshold = torch.tensor(0.1)
    compression = torch.tensor(100.0)
    out = fat_exp_relu(x_tens, threshold, compression)


def test_convert_relus_to_fat():
    model = MLPNet()
    convert_relus_to_fat(model)

    for name, mod in model.named_modules():
        if "act" in name:
            assert isinstance(mod, FATReLU)


def test_set_relu_to_fat():
    model = MLPNet()
    layer_desc = MLPNet.layer_descs()[1]

    set_relu_to_fat(model, layer_desc.name)

    for name, mod in model.named_modules():
        if name == layer_desc.name:
            assert isinstance(mod, FATReLU)
