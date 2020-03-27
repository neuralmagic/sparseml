import pytest

from neuralmagicML.pytorch.models.image_classification.mobilenet_v2 import (
    mobilenet_v2_100,
)

from neuralmagicML.pytorch.models.image_classification.mobilenet import mobilenet


def mostly_not_equal(model_1, model_2):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return True

    return False


def test_mobilenet_v2():
    default_mobilenet = mobilenet_v2_100()
    pretrained_mobilenet = mobilenet_v2_100(pretrained=True)

    assert mostly_not_equal(default_mobilenet, pretrained_mobilenet)


def test_mobilenet():
    default_mobilenet = mobilenet()
    pretrained_mobilenet = mobilenet(pretrained=True)

    assert mostly_not_equal(default_mobilenet, pretrained_mobilenet)


def test_mobilenet_recal():
    default_mobilenet = mobilenet()
    pretrained_mobilenet = mobilenet(pretrained="imagenet/pytorch/recal")

    assert mostly_not_equal(default_mobilenet, pretrained_mobilenet)


def test_mobilenet_recal_perf():
    default_mobilenet = mobilenet()
    pretrained_mobilenet = mobilenet(pretrained="imagenet/pytorch/recal-perf")

    assert mostly_not_equal(default_mobilenet, pretrained_mobilenet)
