import pytest

from typing import Optional, Callable
import sys
import torch
from torch.nn import Module
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from neuralmagicML.pytorch.recal import (
    ScheduledOptimizer,
    ScheduledModifierManager,
)

from .test_modifier import (
    ModifierTest,
    def_model,
    def_optim_sgd,
    def_optim_adam,
    test_loss,
    test_epoch,
    test_steps_per_epoch,
)


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


def test_optim_all_info():
    model = def_model()
    optim = FakeOptim(model.parameters(), 0.1)
    steps_per_epoch = 100
    manager = FakeManager()
    optim = ScheduledOptimizer(optim, model, manager, steps_per_epoch)

    with pytest.raises(RuntimeError):
        optim.epoch_end()

    for epoch in range(10):
        optim.epoch_start()

        for batch in range(steps_per_epoch):
            optim.loss_update(torch.tensor(0.0))
            optim.step()
            expected_epoch = float(epoch) + float(batch) / float(steps_per_epoch)
            assert (
                abs(expected_epoch - manager.last_called_epoch) < sys.float_info.epsilon
            )

        optim.epoch_end()


def test_optim_size_only():
    model = def_model()
    optim = FakeOptim(model.parameters(), 0.1)
    steps_per_epoch = 100
    manager = FakeManager()
    optim = ScheduledOptimizer(optim, model, manager, steps_per_epoch)

    for epoch in range(10):
        for batch in range(steps_per_epoch):
            optim.loss_update(torch.tensor(0.0))
            optim.step()
            expected_epoch = float(epoch) + float(batch) / float(steps_per_epoch)
            assert (
                abs(expected_epoch - manager.last_called_epoch) < sys.float_info.epsilon
            )

    with pytest.raises(RuntimeError):
        optim.epoch_start()

    with pytest.raises(RuntimeError):
        optim.epoch_end()


def test_optim_start_end_only():
    model = def_model()
    optim = FakeOptim(model.parameters(), 0.1)
    steps_per_epoch = 100
    manager = FakeManager()
    optim = ScheduledOptimizer(optim, model, manager, steps_per_epoch=-1)

    with pytest.raises(RuntimeError):
        optim.step()

    for epoch in range(10):
        optim.epoch_start()

        for batch in range(steps_per_epoch):
            optim.loss_update(torch.tensor(0.0))
            optim.step()
            expected_epoch = float(epoch)
            assert (
                abs(expected_epoch - manager.last_called_epoch) < sys.float_info.epsilon
            )

        optim.epoch_end()
