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
from abc import ABC
from typing import Dict

import pytest
import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential

from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    Accuracy,
    BinaryCrossEntropyLossWrapper,
    CrossEntropyLossWrapper,
    KDLossWrapper,
    KDSettings,
    LossWrapper,
    SSDLossWrapper,
    TopKAccuracy,
    YoloLossWrapper,
)


def default_loss_fn(pred: Tensor, lab: Tensor):
    return torch.norm(pred - lab.float(), dim=0)


def default_metric_one(pred: Tensor, lab: Tensor):
    return torch.sum(pred == lab.float(), dim=0)


def default_metric_two(pred: Tensor, lab: Tensor):
    return torch.sum(pred != lab.float(), dim=0)


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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
class LossWrapperTest(ABC):
    def test_available_losses(self, wrapper, metrics, data, pred):
        assert wrapper.available_losses == (DEFAULT_LOSS_KEY, *metrics.keys())

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

    def test_get_labels(self, wrapper, metrics, data, pred):
        if isinstance(data, Tensor):
            with pytest.raises(TypeError):
                wrapper.get_labels(data, pred, DEFAULT_LOSS_KEY)
        else:
            assert (
                torch.sum(
                    (
                        data[1].float()
                        - wrapper.get_labels(data, pred, DEFAULT_LOSS_KEY).float()
                    ).abs()
                )
                < 0.00001
            )

            with pytest.raises(TypeError):
                wrapper.get_labels(data[1], pred, DEFAULT_LOSS_KEY)

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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "wrapper,metrics",
    [
        (BinaryCrossEntropyLossWrapper(DEFAULT_METRICS), DEFAULT_METRICS),
        (
            BinaryCrossEntropyLossWrapper(DEFAULT_METRICS),
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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "wrapper,metrics", [(CrossEntropyLossWrapper(DEFAULT_METRICS), DEFAULT_METRICS)]
)
@pytest.mark.parametrize(
    "data", [(torch.randn(8, 16), torch.ones(8).type(torch.int64)), torch.randn(8, 16)]
)
@pytest.mark.parametrize("pred", [(torch.randn(8, 8), torch.randn(8, 8))])
class TestCrossEntropyLossWrapperImpl(LossWrapperTest):
    pass


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "wrapper,metrics",
    [(SSDLossWrapper({}), {})],
)
@pytest.mark.parametrize(
    "data",
    [
        (
            torch.randn(8, 3, 300, 300),
            (
                torch.randn(8, 4, 8732),
                torch.randint(21, (8, 8732), dtype=torch.long),
                [],
            ),
        )
    ],
)
@pytest.mark.parametrize("pred", [(torch.randn(8, 4, 8732), torch.randn(8, 21, 8732))])
class TestSSDLossWrapperImpl(LossWrapperTest):
    def test_get_preds(self, wrapper, metrics, data, pred):
        preds = wrapper.get_preds(data, pred, DEFAULT_LOSS_KEY)
        assert torch.sum((pred[0] - preds[0]).abs()) < 0.00001
        assert torch.sum((pred[1] - preds[1]).abs()) < 0.00001

    def test_get_labels(self, wrapper, metrics, data, pred):
        labels = wrapper.get_labels(data, pred, DEFAULT_LOSS_KEY)
        assert torch.sum((data[1][0] - labels[0]).abs()) < 0.00001
        assert torch.sum((data[1][1].float() - labels[1].float()).abs()) < 0.00001


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "wrapper,metrics",
    [(YoloLossWrapper({}), {})],
)
@pytest.mark.parametrize(
    "data",
    [
        (
            torch.randn(4, 3, 640, 640),
            (
                torch.cat(
                    (
                        torch.arange(4)[:, None].float(),
                        torch.rand(4, 4),
                        torch.randint(80, (4, 1)).float(),
                    ),
                    1,
                ),
                [],
            ),
        )
    ],
)
@pytest.mark.parametrize(
    "pred",
    [
        (
            torch.randn(4, 3, 20, 20, 85),
            torch.randn(4, 3, 40, 40, 85),
            torch.randn(4, 3, 80, 80, 85),
        )
    ],
)
class TestYoloLossWrapperImpl(LossWrapperTest):
    def test_get_preds(self, wrapper, metrics, data, pred):
        preds = wrapper.get_preds(data, pred, DEFAULT_LOSS_KEY)
        assert torch.sum((pred[0] - preds[0]).abs()) < 0.00001
        assert torch.sum((pred[1] - preds[1]).abs()) < 0.00001
        assert torch.sum((pred[2] - preds[2]).abs()) < 0.00001

    def test_get_labels(self, wrapper, metrics, data, pred):
        labels = wrapper.get_labels(data, pred, DEFAULT_LOSS_KEY)
        assert torch.sum((data[1][0] - labels[0]).abs()) < 0.00001


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
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
def test_accuracy(pred, lab, expected_acc):
    acc = Accuracy.calculate(pred, lab)
    assert torch.sum((acc - Accuracy()(pred, lab)).abs()) < 0.0000001
    assert torch.sum((expected_acc - acc).abs()) < 0.001


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
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
def test_topk_accuracy(pred, lab, topk, expected_acc):
    acc = TopKAccuracy.calculate(pred, lab, topk)
    assert torch.sum((acc - TopKAccuracy(topk)(pred, lab)).abs()) < 0.0000001
    assert torch.sum((expected_acc - acc).abs()) < 0.001
