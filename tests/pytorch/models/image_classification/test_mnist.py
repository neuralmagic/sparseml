from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Conv2d, Linear, BatchNorm2d

from neuralmagicML.pytorch.models import mnist_net


def mostly_not_equal(model_1, model_2):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return True

    return False


def test_mnist_arch():
    model = mnist_net()
    conv_count = 0
    bn_count = 0
    fc_count = 0

    for name, mod in model.named_modules():
        if isinstance(mod, Conv2d):
            conv_count += 1

        if isinstance(mod, BatchNorm2d):
            bn_count += 1

        if isinstance(mod, Linear):
            fc_count += 1

    assert conv_count == 4
    assert bn_count == 4
    assert fc_count == 1


def test_mnist_forward():
    model = mnist_net()
    batch = torch.randn(1, 1, 28, 28)
    out = model(batch)

    assert not isinstance(out, Tensor)
    assert isinstance(out, Tuple)
    assert len(out) == 2
    assert out[0].shape[0] == 1
    assert out[1].shape[0] == 1
    assert out[0].shape[1] == 10
    assert out[1].shape[1] == 10


def test_mnist_pretrained():
    model = mnist_net()
    pretrained = mnist_net(pretrained=True)

    mostly_not_equal(model, pretrained)
