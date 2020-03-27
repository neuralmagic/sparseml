import pytest

from neuralmagicML.pytorch.models.image_classification.vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)


def mostly_not_equal(model_1, model_2):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return True

    return False


def test_vgg11():
    default_vgg = vgg11()
    pretrained_vgg = vgg11(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg11_bn():
    default_vgg = vgg11_bn()
    pretrained_vgg = vgg11_bn(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg13():
    default_vgg = vgg13()
    pretrained_vgg = vgg13(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg13_bn():
    default_vgg = vgg13_bn()
    pretrained_vgg = vgg13_bn(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg16():
    default_vgg = vgg16()
    pretrained_vgg = vgg16(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg16_recal():
    default_vgg = vgg16()
    pretrained_vgg = vgg16(pretrained="imagenet/pytorch/recal")

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg16_recal_perf():
    default_vgg = vgg16()
    pretrained_vgg = vgg16(pretrained="imagenet/pytorch/recal-perf")

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg16_bn():
    default_vgg = vgg16_bn()
    pretrained_vgg = vgg16_bn(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg19():
    default_vgg = vgg19()
    pretrained_vgg = vgg19(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)


def test_vgg19_bn():
    default_vgg = vgg19_bn()
    pretrained_vgg = vgg19_bn(pretrained=True)

    assert mostly_not_equal(default_vgg, pretrained_vgg)
