import pytest

from neuralmagicML.pytorch.models.image_classification.resnet import (
    resnet50,
    resnet50_v2,
    resnet50_2xwidth,
    resnext50,
    resnet101,
    resnet101_v2,
    resnet101_2xwidth,
    resnext101,
    resnet152,
    resnet152_v2,
    resnext152,
)


def mostly_not_equal(model_1, model_2):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return True

    return False


def test_resnet_50():
    default_resnet = resnet50()
    pretrained_resnet = resnet50(pretrained=True)

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnet_50_recal():
    default_resnet = resnet50()
    pretrained_resnet = resnet50(pretrained="imagenet/pytorch/recal")

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnet_50_recal_perf():
    default_resnet = resnet50()
    pretrained_resnet = resnet50(pretrained="imagenet/pytorch/recal-perf")

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnet50_2xwidth():
    default_resnet = resnet50_2xwidth()
    pretrained_resnet = resnet50_2xwidth(pretrained=True)

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnext_50():
    default_resnet = resnext50()
    pretrained_resnet = resnext50(pretrained=True)

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnet101():
    default_resnet = resnet101()
    pretrained_resnet = resnet101(pretrained=True)

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnet101_2xwidth():
    default_resnet = resnet101_2xwidth()
    pretrained_resnet = resnet101_2xwidth(pretrained=True)

    assert mostly_not_equal(default_resnet, pretrained_resnet)


def test_resnet152():
    default_resnet = resnet152()
    pretrained_resnet = resnet152(pretrained=True)

    assert mostly_not_equal(default_resnet, pretrained_resnet)
