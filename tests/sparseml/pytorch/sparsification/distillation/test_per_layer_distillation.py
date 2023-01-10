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
import re
from collections import OrderedDict
from typing import Callable

import pytest
import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import Optimizer

from sparseml.pytorch.models.classification.resnet import resnet18
from sparseml.pytorch.sparsification import (
    Modifier,
    PerLayerDistillationModifier,
    ScheduledModifier,
)
from sparseml.pytorch.sparsification.distillation.modifier_per_layer import (
    DISTILL_PARAM_GROUP_KEY,
)
from sparseml.pytorch.sparsification.quantization.modifier_quantization import (
    QuantizationModifier,
)
from tests.sparseml.pytorch.helpers import create_optim_sgd
from tests.sparseml.pytorch.sparsification.test_modifier import ScheduledModifierTest


# NOTE: these are fixtures used in testing below
from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


def mlp(*layer_sizes: int):
    layers = []
    for idx, size in enumerate(layer_sizes[:-1]):
        layers.append(
            nn.Sequential(
                OrderedDict(
                    [
                        ("linear", nn.Linear(size, layer_sizes[idx + 1])),
                        ("activation", nn.ReLU()),
                    ]
                )
            )
        )
    return nn.Sequential(*layers)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_different_structure_raises_error():
    modifier = PerLayerDistillationModifier()
    student = mlp(12, 24, 32)
    teacher = mlp(8, 12, 24, 64)

    with pytest.raises(
        ValueError,
        match="Found different numbers of teacher and student layers to distill.",
    ):
        modifier.initialize(student, distillation_teacher=teacher)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_different_sizes_without_projection_raises_error():
    modifier = PerLayerDistillationModifier(project_features=False)
    student = mlp(12, 24, 32)
    teacher = mlp(12, 12, 24)
    opt = create_optim_sgd(student)

    modifier.initialize(student, distillation_teacher=teacher)

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The size of tensor a (24) must match "
            "the size of tensor b (12) at non-singleton dimension 1"
        ),
    ):
        x = torch.randn(5, 12)
        fake_loss = student(x).mean()
        modifier.loss_update(
            fake_loss,
            student,
            opt,
            modifier.start_epoch,
            10,
            student_outputs=fake_loss,
            student_inputs=x,
        )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_same_structure_different_layer_sizes():
    modifier = PerLayerDistillationModifier(
        student_layer_names=["1.linear", "2.linear"],
        teacher_layer_names=["1.linear", "2.linear"],
        project_features=True,
    )
    student = mlp(12, 24, 32, 64)
    teacher = mlp(12, 48, 64, 128)
    opt = create_optim_sgd(student)
    modifier.initialize(student, distillation_teacher=teacher)

    x = torch.randn(5, 12)
    fake_loss = student(x).mean()
    updated_loss = modifier.loss_update(
        fake_loss,
        student,
        opt,
        modifier.start_epoch,
        10,
        student_inputs=x,
        student_outputs=fake_loss,
    )

    assert isinstance(updated_loss, torch.Tensor)
    assert updated_loss.shape == fake_loss.shape
    assert fake_loss.item() != updated_loss.item()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_different_structure():
    modifier = PerLayerDistillationModifier(
        student_layer_names=["0.linear", "1.linear"],
        teacher_layer_names=["1.linear", "2.linear"],
        project_features=True,
    )
    student = mlp(12, 24, 32)
    teacher = mlp(12, 12, 24, 64)
    opt = create_optim_sgd(student)
    modifier.initialize(student, distillation_teacher=teacher)

    x = torch.randn(5, 12)
    fake_loss = student(x).mean()
    updated_loss = modifier.loss_update(
        fake_loss,
        student,
        opt,
        modifier.start_epoch,
        10,
        student_inputs=x,
        student_outputs=fake_loss,
    )

    assert isinstance(updated_loss, torch.Tensor)
    assert updated_loss.shape == fake_loss.shape
    assert fake_loss.item() != updated_loss.item()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda,model_lambda",
    [
        (
            # same structure & sizes
            lambda: PerLayerDistillationModifier(
                student_layer_names=["0.linear", "1.linear"],
                teacher_layer_names=["0.linear", "1.linear"],
                project_features=False,
            ),
            lambda: mlp(12, 24, 32),
        ),
    ],
)
@pytest.mark.parametrize("optim_lambda", [create_optim_sgd])
class TestPerLayerDistillationModifierImpl(ScheduledModifierTest):
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

        student_inputs = torch.randn(5, 12)
        student_outputs = model(student_inputs)
        fake_loss = student_outputs.mean()
        updated_loss = modifier.loss_update(
            fake_loss,
            model,
            optimizer,
            modifier.start_epoch,
            test_steps_per_epoch,
            student_inputs=student_inputs,
            student_outputs=student_outputs,
        )

        assert isinstance(updated_loss, torch.Tensor)
        assert updated_loss.shape == fake_loss.shape
        assert fake_loss.item() != updated_loss.item()


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_distillation_modifier_yaml():
    truth = PerLayerDistillationModifier(
        gain=2.0,
        start_epoch=0.0,
        end_epoch=12.0,
        update_frequency=1.5,
        normalize=False,
        student_layer_names=["a", "b", "c"],
        teacher_layer_names=["d", "e", "f"],
        project_features=False,
        epsilon=3.0,
    )
    yaml_str = f"""
        !PerLayerDistillationModifier
            gain: {truth.gain}
            start_epoch: {truth.start_epoch}
            end_epoch: {truth.end_epoch}
            update_frequency: {truth.update_frequency}
            normalize: {truth.normalize}
            student_layer_names: {truth.student_layer_names}
            teacher_layer_names: {truth.teacher_layer_names}
            project_features: {truth.project_features}
            epsilon: {truth.epsilon}
        """
    from_yaml = PerLayerDistillationModifier.load_obj(yaml_str)
    twice_from_yaml = PerLayerDistillationModifier.load_obj(str(from_yaml))

    assert isinstance(from_yaml, PerLayerDistillationModifier)
    for attr_name in [
        "gain",
        "start_epoch",
        "end_epoch",
        "update_frequency",
        "normalize",
        "student_layer_names",
        "teacher_layer_names",
        "project_features",
        "epsilon",
    ]:
        assert (
            getattr(truth, attr_name)
            == getattr(from_yaml, attr_name)
            == getattr(twice_from_yaml, attr_name)
        )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_optimizer_serialization_with_projections(tmp_path):
    # since we are adding param groups to optimizer during initialization, we need
    # to ensure that they can be serialized/reloaded properly
    optim_checkpoint_path = tmp_path / "optim.pth"
    student = mlp(12, 24, 64)
    teacher = mlp(12, 24, 64)

    modifier = PerLayerDistillationModifier()
    modifier.initialize(module=student, distillation_teacher=teacher)

    optimizer = create_optim_sgd(student)
    x = torch.randn(5, 12)
    fake_loss = student(x).mean()

    # add state to optimizer
    modifier.loss_update(
        loss=fake_loss,
        module=student,
        optimizer=optimizer,
        epoch=modifier.start_epoch,
        steps_per_epoch=10,
        student_inputs=x,
        student_outputs=fake_loss,
    )

    optimizer_checkpoint = {
        "optimizer": optimizer.state_dict(),
    }
    torch.save(optimizer_checkpoint, optim_checkpoint_path)
    loaded_optimizer_checkpoint = torch.load(optim_checkpoint_path)

    loaded_optimizer = create_optim_sgd(student)
    loaded_optimizer.load_state_dict(loaded_optimizer_checkpoint["optimizer"])

    # also check that we can reload the state dict into our optimizer
    optimizer.load_state_dict(loaded_optimizer_checkpoint["optimizer"])


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_modifier_load_before_forward_pass(tmp_path):
    student = mlp(12, 24, 64)
    teacher = mlp(12, 24, 64)

    modifier = PerLayerDistillationModifier()
    modifier.initialize(module=student, distillation_teacher=teacher)

    optimizer = create_optim_sgd(student)
    x = torch.randn(5, 12)
    fake_loss = student(x).mean()

    # add state to optimizer
    modifier.loss_update(
        loss=fake_loss,
        module=student,
        optimizer=optimizer,
        epoch=modifier.start_epoch,
        steps_per_epoch=10,
        student_inputs=x,
        student_outputs=fake_loss,
    )

    modifier_2 = PerLayerDistillationModifier()
    modifier_2.initialize(module=student, distillation_teacher=teacher)
    modifier_2.load_state_dict(modifier.state_dict())

    # call forward again
    student(x)
    modifier_2.loss_update(
        loss=fake_loss,
        module=student,
        optimizer=optimizer,
        epoch=modifier.start_epoch,
        steps_per_epoch=10,
        student_inputs=x,
        student_outputs=fake_loss,
    )
    assert DISTILL_PARAM_GROUP_KEY in modifier.state_dict()
    assert DISTILL_PARAM_GROUP_KEY in modifier_2.state_dict()
    assert _check_state_dict_equality(modifier.state_dict(), modifier_2.state_dict())


def _check_state_dict_equality(expected_dict, actual_dict):
    assert len(expected_dict) == len(actual_dict)

    for ((expected_key, expected_value), (actual_key, actual_value)) in zip(
        expected_dict.items(), actual_dict.items()
    ):
        assert expected_key == actual_key
        assert type(expected_value) == type(actual_value)

        if isinstance(expected_value, (dict, OrderedDict)):
            assert _check_state_dict_equality(expected_value, actual_value)
        else:
            # if not dict type the values are always Tensors
            assert torch.allclose(expected_value, actual_value)
    return True


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_hooks_survive_after_quantization_is_applied():
    # this should test fused module hooks are properly kept around during
    # quantization
    modifier = PerLayerDistillationModifier()
    student = resnet18(pretrained=True)
    teacher = resnet18(pretrained=True)
    opt = create_optim_sgd(student)

    modifier.initialize(student, distillation_teacher=teacher)

    # now apply quantization - hooks may disappear here
    QuantizationModifier().apply_structure(student)

    x = torch.rand(2, 3, 224, 224)
    student_outputs = student(x)
    fake_loss = student_outputs[0].square().mean()
    updated_loss = modifier.loss_update(
        fake_loss,
        student,
        opt,
        modifier.start_epoch,
        10,
        student_inputs=x,
        student_outputs=student_outputs,
    )

    assert isinstance(updated_loss, torch.Tensor)
    assert updated_loss.shape == fake_loss.shape
    assert fake_loss.item() != updated_loss.item()
