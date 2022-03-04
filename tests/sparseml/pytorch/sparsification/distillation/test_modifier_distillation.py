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
from typing import Callable

import pytest
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from sparseml.pytorch.sparsification import (
    DistillationModifier,
    Modifier,
    ScheduledModifier,
)
from tests.sparseml.pytorch.helpers import LinearNet, create_optim_sgd
from tests.sparseml.pytorch.sparsification.test_modifier import ScheduledModifierTest


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


DISTILLATION_MODIFIERS = [
    lambda: DistillationModifier(start_epoch=0.0),
]


def _get_fake_batch(model_lambda):
    batch_size = 5
    input_shape = model_lambda.layer_descs()[0].input_size
    return torch.randn(batch_size, *input_shape)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("modifier_lambda", DISTILLATION_MODIFIERS, scope="function")
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd], scope="function")
class TestDistillationModifierImpl(ScheduledModifierTest):
    def test_update_ready(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
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
            distillation_teacher=model_lambda(),
        )

    def test_scheduled_update(
        self,
        modifier_lambda: Callable[[], ScheduledModifier],
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
            distillation_teacher=model_lambda(),
        )

    def test_lifecycle(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)

        self.initialize_helper(modifier, model, distillation_teacher=model_lambda())

        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)

        assert modifier.update_ready(modifier.start_epoch, test_steps_per_epoch)
        modifier.scheduled_update(
            model, optimizer, modifier.start_epoch, test_steps_per_epoch
        )

        if modifier.end_epoch > modifier.start_epoch:
            assert not modifier.update_ready(
                (modifier.start_epoch + modifier.end_epoch) / 2, test_steps_per_epoch
            )
            assert modifier.update_ready(modifier.end_epoch, test_steps_per_epoch)

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

        self.initialize_helper(modifier, model, distillation_teacher=model_lambda())

        # test distillation has been applied
        # fake forward pass
        student_inputs = _get_fake_batch(model_lambda)
        student_outputs = model(student_inputs)
        fake_loss = student_outputs.mean()
        updated_loss = modifier.loss_update(
            fake_loss,
            model,
            optimizer,
            modifier.start_epoch,
            test_steps_per_epoch,
            student_outputs,
            student_inputs,
        )

        assert isinstance(updated_loss, torch.Tensor)
        assert updated_loss.shape == fake_loss.shape
        assert fake_loss.item() != updated_loss.item()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_distillation_modifier_yaml():
    start_epoch = 0.0
    hardness = 0.9
    temperature = 5.0
    distill_output_keys = [0]
    yaml_str = f"""
        !DistillationModifier
            start_epoch: {start_epoch}
            hardness: {hardness}
            temperature: {temperature}
            distill_output_keys: {distill_output_keys}
        """
    yaml_modifier = DistillationModifier.load_obj(
        yaml_str
    )  # type: DistillationModifier
    serialized_modifier = DistillationModifier.load_obj(
        str(yaml_modifier)
    )  # type: DistillationModifier
    obj_modifier = DistillationModifier(
        start_epoch=start_epoch,
        hardness=hardness,
        temperature=temperature,
        distill_output_keys=distill_output_keys,
    )

    assert isinstance(yaml_modifier, DistillationModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.hardness == serialized_modifier.hardness == obj_modifier.hardness
    )
    assert (
        yaml_modifier.temperature
        == serialized_modifier.temperature
        == obj_modifier.temperature
    )
    assert (
        yaml_modifier.distill_output_keys
        == serialized_modifier.distill_output_keys
        == obj_modifier.distill_output_keys
    )
