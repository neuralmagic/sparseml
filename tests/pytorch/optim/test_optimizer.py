import pytest

import os
from typing import Optional, Callable
import sys
import torch
from torch.nn import Module
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.optim import (
    ScheduledOptimizer,
    ScheduledModifierManager,
)

from tests.pytorch.helpers import MLPNet


class FakeOptim(SGD):
    def zero_grad(self) -> None:
        return

    def step(self, closure: Optional[Callable[[], float]] = ...) -> None:
        return


class FakeManager(ScheduledModifierManager):
    def __init__(self):
        super().__init__([])
        self.last_called_epoch = -1

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        super().update(module, optimizer, epoch, steps_per_epoch)
        self.last_called_epoch = epoch


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_optim():
    model = MLPNet()
    optim = FakeOptim(model.parameters(), 0.1)
    steps_per_epoch = 100
    manager = FakeManager()

    with pytest.raises(ValueError):
        ScheduledOptimizer(optim, model, manager, steps_per_epoch=-1)

    optim = ScheduledOptimizer(optim, model, manager, steps_per_epoch)

    for epoch in range(10):
        for batch in range(steps_per_epoch):
            optim.loss_update(torch.tensor(0.0))
            optim.step()
            expected_epoch = float(epoch) + float(batch) / float(steps_per_epoch)
            assert (
                abs(expected_epoch - manager.last_called_epoch) < sys.float_info.epsilon
            )
