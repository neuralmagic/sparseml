import pytest

from abc import ABC
from typing import Dict, Iterable
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU

from neuralmagicML.utils import (
    DEFAULT_LOSS_KEY,
    TEACHER_LOSS_KEY,
    LossWrapper,
    KDSettings,
    KDLossWrapper,
    BinaryCrossEntropyLossWrapper,
    CrossEntropyLossWrapper,
    Accuracy,
    TopKAccuracy,
)


def default_loss_fn(pred: Tensor, lab: Tensor):
    return torch.norm(pred - lab, dim=0)


def default_metric_one(pred: Tensor, lab: Tensor):
    return torch.sum(pred == lab, dim=0)


def default_metric_two(pred: Tensor, lab: Tensor):
    return torch.sum(pred != lab, dim=0)


def default_kd_parent():
    return Sequential(
        Linear(16, 32),
        ReLU(),
        Linear(32, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 2),
        ReLU(),
    )


DEFAULT_METRICS = {"one": default_metric_one, "two": default_metric_two}
DEFAULT_KD_SETTINGS = KDSettings(teacher=default_kd_parent())


class LossWrapperTest(ABC):
    @pytest.mark.device_cpu
    def test_available_losses(self, wrapper, metrics, data, pred):
        assert wrapper.available_losses == (DEFAULT_LOSS_KEY, *metrics.keys())

    @pytest.mark.device_cpu
    def test_get_inputs(self, wrapper, metrics, data, pred):
        if isinstance(data, Tensor):
            assert (
                torch.sum(
                    (data - wrapper.get_inputs(data, pred, DEFAULT_LOSS_KEY)).abs()
                )
                < 0.00001
            )
        else:
            assert (
                torch.sum(
                    (data[0] - wrapper.get_inputs(data, pred, DEFAULT_LOSS_KEY)).abs()
                )
                < 0.00001
            )
            assert (
                torch.sum(
                    (
                        data[0] - wrapper.get_inputs(data[0], pred, DEFAULT_LOSS_KEY)
                    ).abs()
                )
                < 0.00001
            )

    @pytest.mark.device_cpu
    def test_get_preds(self, wrapper, metrics, data, pred):
        if isinstance(pred, Tensor):
            assert (
                torch.sum(
                    (pred - wrapper.get_preds(data, pred, DEFAULT_LOSS_KEY)).abs()
                )
                < 0.00001
            )
        else:
            assert (
                torch.sum(
                    (pred[0] - wrapper.get_preds(data, pred, DEFAULT_LOSS_KEY)).abs()
                )
                < 0.00001
            )
            assert (
                torch.sum(
                    (pred[0] - wrapper.get_preds(data, pred[0], DEFAULT_LOSS_KEY)).abs()
                )
                < 0.00001
            )

    @pytest.mark.device_cpu
    def test_get_labels(self, wrapper, metrics, data, pred):
        if isinstance(data, Tensor):
            with pytest.raises(TypeError):
                wrapper.get_labels(data, pred, DEFAULT_LOSS_KEY)
        else:
            assert (
                torch.sum(
                    (data[1] - wrapper.get_labels(data, pred, DEFAULT_LOSS_KEY)).abs()
                )
                < 0.00001
            )

            with pytest.raises(TypeError):
                wrapper.get_labels(data[1], pred, DEFAULT_LOSS_KEY)

    @pytest.mark.device_cpu
    def test_calc_loss(self, wrapper, metrics, data, pred):
        if isinstance(data, Tensor):
            return

        pred_inp = pred[0] if isinstance(pred, Iterable) else pred
        assert isinstance(wrapper.calc_loss(data[0], pred_inp, data[1]), Tensor)

    @pytest.mark.device_cpu
    def test_forward(self, wrapper, metrics, data, pred):
        if isinstance(data, Tensor):
            with pytest.raises(TypeError):
                wrapper.forward(data, pred)

            return

        losses = wrapper.forward(data, pred)
        assert isinstance(losses, Dict)
        assert DEFAULT_LOSS_KEY in losses

        for key in metrics.keys():
            assert key in losses

    @pytest.mark.device_cpu
    def test_callable(self, wrapper, metrics, data, pred):
        if isinstance(data, Tensor):
            with pytest.raises(TypeError):
                wrapper.forward(data, pred)

            return

        losses = wrapper.forward(data, pred)
        assert isinstance(losses, Dict)
        assert DEFAULT_LOSS_KEY in losses

        for key in metrics.keys():
            assert key in losses


@pytest.mark.parametrize(
    "wrapper,metrics",
    [(LossWrapper(default_loss_fn, DEFAULT_METRICS), DEFAULT_METRICS)],
)
@pytest.mark.parametrize(
    "data",
    [(torch.randn(8, 3, 224, 224), torch.randn(8, 1)), torch.randn(8, 3, 224, 224)],
)
@pytest.mark.parametrize("pred", [(torch.randn(8), torch.randn(8)), torch.randn(8)])
class TestLossWrapperImpl(LossWrapperTest):
    pass


@pytest.mark.parametrize(
    "wrapper,metrics",
    [
        (KDLossWrapper(default_loss_fn, DEFAULT_METRICS), DEFAULT_METRICS),
        (
            KDLossWrapper(
                default_loss_fn, DEFAULT_METRICS, kd_settings=DEFAULT_KD_SETTINGS
            ),
            DEFAULT_METRICS,
        ),
    ],
)
@pytest.mark.parametrize(
    "data", [(torch.randn(8, 16), torch.randn(8, 2)), torch.randn(8, 16)]
)
@pytest.mark.parametrize("pred", [(torch.randn(8, 2), torch.randn(8, 2))])
class TestKDLossWrapperImpl(LossWrapperTest):
    pass


@pytest.mark.parametrize(
    "wrapper,metrics",
    [
        (BinaryCrossEntropyLossWrapper(DEFAULT_METRICS), DEFAULT_METRICS),
        (
            BinaryCrossEntropyLossWrapper(
                DEFAULT_METRICS, kd_settings=DEFAULT_KD_SETTINGS
            ),
            DEFAULT_METRICS,
        ),
    ],
)
@pytest.mark.parametrize(
    "data", [(torch.randn(8, 16), torch.randn(8, 2)), torch.randn(8, 16)]
)
@pytest.mark.parametrize("pred", [(torch.randn(8, 2), torch.randn(8, 2))])
class TestBinaryCrossEntropyLossWrapperImpl(LossWrapperTest):
    pass


@pytest.mark.parametrize(
    "wrapper,metrics", [(CrossEntropyLossWrapper(DEFAULT_METRICS), DEFAULT_METRICS)]
)
@pytest.mark.parametrize(
    "data", [(torch.randn(8, 16), torch.ones(8).type(torch.int64)), torch.randn(8, 16)]
)
@pytest.mark.parametrize("pred", [(torch.randn(8, 8), torch.randn(8, 8))])
class TestCrossEntropyLossWrapperImpl(LossWrapperTest):
    pass


@pytest.mark.parametrize(
    "pred,lab,expected_acc",
    [
        (torch.zeros(8, 2), torch.ones(8, 2), 0.0),
        (torch.zeros(8, 2), torch.zeros(8, 2), 100),
        (torch.ones(16), torch.ones(16), 100),
        (
            torch.tensor([1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9]),
            torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
            50,
        ),
    ],
)
@pytest.mark.device_cpu
def test_accuracy(pred, lab, expected_acc):
    acc = Accuracy.calculate(pred, lab)
    assert torch.sum((acc - Accuracy()(pred, lab)).abs()) < 0.0000001
    assert torch.sum((expected_acc - acc).abs()) < 0.001


@pytest.mark.parametrize(
    "pred,lab,topk,expected_acc",
    [
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                ]
            ),
            torch.tensor([[4], [7]]),
            1,
            0.0,
        ),
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                ]
            ),
            torch.tensor([[4], [7]]),
            2,
            50.0,
        ),
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                ]
            ),
            torch.tensor([[4], [7]]),
            4,
            100.0,
        ),
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                    [1.0, 0.0, 0.8, 0.5, 0.8, 0.4, 0.6, 0.9],
                ]
            ),
            torch.tensor([[4], [7]]),
            5,
            100.0,
        ),
    ],
)
@pytest.mark.device_cpu
def test_topk_accuracy(pred, lab, topk, expected_acc):
    acc = TopKAccuracy.calculate(pred, lab, topk)
    assert torch.sum((acc - TopKAccuracy(topk)(pred, lab)).abs()) < 0.0000001
    assert torch.sum((expected_acc - acc).abs()) < 0.001
