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
import sys
from typing import Callable, Optional

import pytest
import torch
from torch.nn import Module
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from tests.sparseml.pytorch.helpers import MLPNet


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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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
