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
from typing import Callable, List, Union

import pytest
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.optim import (
    PYTORCH_FRAMEWORK,
    Modifier,
    PyTorchModifierYAML,
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import PythonLogger, TensorBoardLogger
from sparseml.utils import ALL_TOKEN
from tests.sparseml.optim import BaseModifierTest, BaseScheduledTest, BaseUpdateTest
from tests.sparseml.pytorch.helpers import (
    LinearNet,
    create_optim_adam,
    create_optim_sgd,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


__all__ = [
    "ModifierTest",
    "ScheduledModifierTest",
    "ScheduledUpdateModifierTest",
    "ModifierImpl",
    "ScheduledModifierImpl",
    "ScheduledUpdateModifierImpl",
]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
class ModifierTest(BaseModifierTest):
    # noinspection PyMethodOverriding
    def initialize_helper(
        self,
        modifier: Modifier,
        model: Module = None,
        epoch: float = 0.0,
        log_initialize: bool = True,
    ):
        modifier.initialize(model, epoch)

        if log_initialize:
            modifier.initialize_loggers([PythonLogger()])

    # noinspection PyMethodOverriding
    def test_constructor(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_constructor(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_yaml(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_yaml_key(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_yaml_key(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_repr(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_repr(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        model = model_lambda()
        super().test_props(
            modifier_lambda,
            framework=PYTORCH_FRAMEWORK,
            initialize_kwargs={"model": model, "epoch": test_epoch},
        )

    def test_initialize(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
    ):
        modifier = modifier_lambda()
        model = model_lambda()

        self.initialize_helper(modifier, model)
        assert modifier.initialized

    def test_initialize_loggers(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optim_lambda(model)

        loggers = []
        expected_loggers = []

        if modifier.log_types == ALL_TOKEN or (
            isinstance(modifier.log_types, List) and "python" in modifier.log_types
        ):
            logger = PythonLogger()
            loggers.append(logger)
            expected_loggers.append(logger)

        if modifier.log_types == ALL_TOKEN or (
            isinstance(modifier.log_types, List) and "tensorboard" in modifier.log_types
        ):
            logger = TensorBoardLogger()
            loggers.append(logger)
            expected_loggers.append(logger)

        modifier.initialize_loggers(loggers)
        assert len(expected_loggers) == len(modifier.loggers)

        for logger in loggers:
            assert logger in modifier.loggers

    def test_update(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)

        self.initialize_helper(modifier, model)

        modifier.enabled = False
        with pytest.raises(RuntimeError):
            modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)
        modifier.enabled = True

        modifier.update(model, optimizer, test_epoch, test_steps_per_epoch)

    def test_log_update(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.log_update(model, optimizer, test_epoch, test_steps_per_epoch)

        self.initialize_helper(modifier, model, log_initialize=True)

        modifier.enabled = False
        with pytest.raises(RuntimeError):
            modifier.log_update(model, optimizer, test_epoch, test_steps_per_epoch)
        modifier.enabled = True

        modifier.log_update(model, optimizer, test_epoch, test_steps_per_epoch)

    def test_loss_update(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
        test_loss: Tensor,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.loss_update(
                test_loss, model, optimizer, test_epoch, test_steps_per_epoch
            )

        self.initialize_helper(modifier, model)
        new_loss = modifier.loss_update(
            test_loss, model, optimizer, test_epoch, test_steps_per_epoch
        )

        assert isinstance(new_loss, Tensor)

    def test_optimizer_pre_step(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.optimizer_pre_step(
                model, optimizer, test_epoch, test_steps_per_epoch
            )

        self.initialize_helper(modifier, model)

        modifier.enabled = False
        with pytest.raises(RuntimeError):
            modifier.optimizer_pre_step(
                model, optimizer, test_epoch, test_steps_per_epoch
            )
        modifier.enabled = True

        modifier.optimizer_pre_step(model, optimizer, test_epoch, test_steps_per_epoch)

    def test_optimizer_post_step(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.optimizer_post_step(
                model, optimizer, test_epoch, test_steps_per_epoch
            )

        self.initialize_helper(modifier, model)

        modifier.enabled = False
        with pytest.raises(RuntimeError):
            modifier.optimizer_post_step(
                model, optimizer, test_epoch, test_steps_per_epoch
            )
        modifier.enabled = True

        modifier.optimizer_post_step(model, optimizer, test_epoch, test_steps_per_epoch)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
class ScheduledModifierTest(ModifierTest, BaseScheduledTest):
    def start_helper(self, modifier: Modifier, model: Module, optimizer: Optimizer):
        modifier._started = True

    # noinspection PyMethodOverriding
    def test_props_start(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_props_start(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    # noinspection PyMethodOverriding
    def test_props_end(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_props_end(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    def test_start_pending(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()

        with pytest.raises(RuntimeError):
            modifier.start_pending(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model)
        modifier.enabled = False
        assert not modifier.start_pending(modifier.start_epoch, test_steps_per_epoch)
        modifier.enabled = True

        if modifier.start_epoch < 0.0:
            assert modifier.start_pending(0.0, test_steps_per_epoch)
        elif modifier.start_epoch > 0.0:
            assert not modifier.start_pending(0.0, test_steps_per_epoch)
            assert modifier.start_pending(modifier.start_epoch, test_steps_per_epoch)

    def test_end_pending(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.end_pending(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model)
        self.start_helper(modifier, model, optimizer)
        modifier.enabled = False
        assert not modifier.end_pending(modifier.start_epoch, test_steps_per_epoch)
        modifier.enabled = True

        if modifier.end_epoch < 0.0:
            assert not modifier.end_pending(modifier.start_epoch, test_steps_per_epoch)
        elif modifier.end_epoch > 0.0:
            assert not modifier.end_pending(0.0, test_steps_per_epoch)
            assert not modifier.end_pending(modifier.start_epoch, test_steps_per_epoch)
            assert modifier.end_pending(modifier.end_epoch, test_steps_per_epoch)

    def test_update_ready(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.update_ready(0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model)
        modifier.enabled = False
        assert not modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
        modifier.enabled = True
        assert modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)

        self.start_helper(modifier, model, optimizer)

        if modifier.end_epoch < 0.0:
            assert not modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
        elif modifier.end_epoch > 0.0:
            assert not modifier.update_ready(0.0, test_steps_per_epoch)
            assert not modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
            assert modifier.update_ready(modifier.end_epoch, test_steps_per_epoch)

    def test_scheduled_update(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.scheduled_update(model, optimizer, 0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model)

        if modifier.start_epoch <= 0.0:
            modifier.scheduled_update(model, optimizer, 0.0, test_steps_per_epoch)
        else:
            with pytest.raises(RuntimeError):
                modifier.scheduled_update(model, optimizer, 0.0, test_steps_per_epoch)

            modifier.scheduled_update(
                model, optimizer, modifier.start_epoch, test_steps_per_epoch
            )

        self.start_helper(modifier, model, optimizer)

        if modifier.end_epoch < 0.0:
            with pytest.raises(RuntimeError):
                modifier.scheduled_update(
                    model, optimizer, modifier.start_epoch, test_steps_per_epoch
                )
        elif modifier.end_epoch > 0.0:
            modifier.scheduled_update(
                model, optimizer, modifier.end_epoch, test_steps_per_epoch
            )

    def test_update(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        with pytest.raises(RuntimeError):
            super().test_update(
                modifier_lambda,
                model_lambda,
                optim_lambda,
                test_epoch,
                test_steps_per_epoch,
            )

    def test_scheduled_log_update(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        with pytest.raises(RuntimeError):
            modifier.scheduled_log_update(model, optimizer, 0.0, test_steps_per_epoch)

        self.initialize_helper(modifier, model, log_initialize=True)

        for epoch in range(
            int(modifier.start_epoch) if modifier.start_epoch >= 0.0 else 0,
            int(modifier.start_epoch) + 5
            if modifier.start_epoch > 0.0
            else int(modifier.start_epoch) + 15,
        ):
            modifier.scheduled_log_update(model, optimizer, 0.0, test_steps_per_epoch)

    def test_log_update(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        with pytest.raises(RuntimeError):
            super().test_log_update(
                modifier_lambda,
                model_lambda,
                optim_lambda,
                test_epoch,
                test_steps_per_epoch,
            )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
class ScheduledUpdateModifierTest(ScheduledModifierTest, BaseUpdateTest):
    # noinspection PyMethodOverriding
    def test_props_frequency(
        self,
        modifier_lambda: Callable[[], Modifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811  # noqa: F811
        test_steps_per_epoch: float,  # noqa: F811
    ):
        super().test_props_frequency(modifier_lambda, framework=PYTORCH_FRAMEWORK)

    def start_helper(
        self, modifier: ScheduledUpdateModifier, model: Module, optimizer: Optimizer
    ):
        super().start_helper(modifier, model, optimizer)
        modifier._last_update_epoch = modifier.start_epoch

    def test_update_ready(
        self,
        modifier_lambda: Callable[[], ScheduledUpdateModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        super().test_update_ready(
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_epoch,
            test_steps_per_epoch,
        )
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model)
        self.start_helper(modifier, model, optimizer)
        min_update_freq = 1.0 / float(test_steps_per_epoch)

        if modifier.update_frequency <= min_update_freq + sys.float_info.epsilon:
            assert modifier.update_ready(
                modifier.start_epoch + min_update_freq, test_steps_per_epoch
            )
        else:
            assert not modifier.update_ready(
                modifier.start_epoch + min_update_freq, test_steps_per_epoch
            )
            assert modifier.update_ready(
                modifier.start_epoch + modifier.update_frequency, test_steps_per_epoch
            )

    def test_scheduled_update(
        self,
        modifier_lambda: Callable[[], ScheduledUpdateModifier],
        model_lambda: Callable[[], Module],
        optim_lambda: Callable[[Module], Optimizer],
        test_epoch: float,  # noqa: F811
        test_steps_per_epoch: int,  # noqa: F811
    ):
        super().test_scheduled_update(
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_epoch,
            test_steps_per_epoch,
        )
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model)
        self.start_helper(modifier, model, optimizer)
        min_update_freq = 1.0 / float(test_steps_per_epoch)

        if modifier.update_frequency <= min_update_freq + sys.float_info.epsilon:
            modifier.scheduled_update(
                model,
                optimizer,
                modifier.start_epoch + min_update_freq,
                test_steps_per_epoch,
            )
        else:
            with pytest.raises(RuntimeError):
                modifier.scheduled_update(
                    model,
                    optimizer,
                    modifier.start_epoch + min_update_freq,
                    test_steps_per_epoch,
                )

            modifier.scheduled_update(
                model,
                optimizer,
                modifier.start_epoch + modifier.update_frequency,
                test_steps_per_epoch,
            )


@PyTorchModifierYAML()
class ModifierImpl(Modifier):
    def __init__(self, log_types: Union[str, List[str]] = ["python"]):
        super().__init__(log_types)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", [ModifierImpl], scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function"
)
class TestModifierImpl(ModifierTest):
    pass


@PyTorchModifierYAML()
class ScheduledModifierImpl(ScheduledModifier):
    def __init__(
        self,
        log_types: Union[str, List[str]] = ["python"],
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
    ):
        super().__init__(log_types)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", [ScheduledModifierImpl], scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function"
)
class TestScheduledModifierImpl(ScheduledModifierTest):
    pass


@PyTorchModifierYAML()
class ScheduledUpdateModifierImpl(ScheduledUpdateModifier):
    def __init__(
        self,
        log_types: Union[str, List[str]] = ["python"],
        end_epoch: float = -1.0,
        start_epoch: float = -1.0,
        update_frequency: float = -1,
    ):
        super().__init__(log_types)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda", [ScheduledUpdateModifierImpl], scope="function"
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function"
)
class TestScheduledUpdateModifierImpl(ScheduledUpdateModifierTest):
    pass


_SAMPLE_RECIPE = """
modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 1.0

  - !SetLearningRateModifier
    start_epoch: 0.0
    learning_rate: 0.1
"""

_SAMPLE_GROUPED_RECIPE = """
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 50.0

  - !SetLearningRateModifier
    start_epoch: 0.0
    learning_rate: 0.1

pruning_modifiers:
  - !GMPruningModifier
    start_epoch: 0
    end_epoch: 40
    init_sparsity: 0.05
    final_sparsity: 0.85
    params: __ALL__
    update_frequency: 0.5
"""


@pytest.mark.parametrize(
    "modifier_str,num_modifiers",
    [
        (_SAMPLE_RECIPE, 2),
        (_SAMPLE_GROUPED_RECIPE, 3),
    ],
)
def test_load_list(modifier_str, num_modifiers):
    modifier_list = Modifier.load_list(modifier_str)
    assert len(modifier_list) == num_modifiers


_SAMPLE_BAD_RECIPE = """
incorrect_modifier_list_name:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 1.0

  - !SetLearningRateModifier
    start_epoch: 0.0
    learning_rate: 0.1
"""


@pytest.mark.parametrize(
    "modifier_str",
    [
        _SAMPLE_BAD_RECIPE,
    ],
)
def test_load_list_fails(modifier_str):
    # expect ValueError mentioning 'modifiers'
    with pytest.raises(ValueError, match=r".*'modifiers'.*"):
        Modifier.load_list(modifier_str)
