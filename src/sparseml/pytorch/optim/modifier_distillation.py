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

"""
Modifier for performing model distillation
"""


from typing import Any, Dict, Iterable, List, Optional

import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from sparseml.optim import ModifierProp
from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.utils import BaseLogger, tensors_module_forward


__all__ = [
    "DistillationModifier",
]


@PyTorchModifierYAML()
class DistillationModifier(ScheduledModifier):
    """
    Adds a knowledge distillation loss based on a teacher model during the
    loss_update phase of the SparseML lifecycle. A distillation_teacher
    module must be provided as a kwarg to the Manager initialization and
    loss_update(loss) must be called before any backwards pass in the integrated
    training flow

    | Sample yaml:
    |   !DistillationModifier
    |       start_epoch: 0.0
    |       hardness: 0.5
    |       temperature: 2.0
    |       distill_output_keys: [0]

    :param start_epoch: The epoch to start the modifier at
    :param hardness: how much to weight the distillation loss vs the base loss
        (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss).
        Default is 0.5
    :param temperature: temperature applied to teacher and student softmax for
        distillation
    :param distill_output_keys: list of keys to of module outputs to use for
        distillation if multiple outputs are present. No or empty list defaults
        to using all available outputs
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        hardness: float = 0.5,
        temperature: float = 2.0,
        distill_output_keys: List[Any] = None,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )
        self._hardness = hardness
        self._temperature = temperature
        self._distill_output_keys = distill_output_keys or []

        self._teacher = None
        self._distillation_enabled = False
        self._track_student_hook = None
        self._student_inputs = None  # last forward inputs to student module
        self._student_outputs = None  # last forward outputs of student module

    @ModifierProp()
    def hardness(self) -> float:
        """
        :return: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        return self._hardness

    @hardness.setter
    def hardness(self, value: float):
        """
        :params value: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        self._hardness = value

    @ModifierProp()
    def temperature(self) -> float:
        """
        :return: temperature applied to teacher and student softmax for
            distillation
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """
        :params value: temperature applied to teacher and student softmax for
            distillation
        """
        self._temperature = value

    @ModifierProp()
    def distill_output_keys(self) -> List[Any]:
        """
        :return: list of keys to of module outputs to use for distillation
            if multiple outputs are present. No or empty list defaults
            to using all available outputs
        """
        return self._distill_output_keys

    @distill_output_keys.setter
    def distill_output_keys(self, value: List[Any]):
        """
        :params value: list of keys to of module outputs to use for distillation
            if multiple outputs are present. No or empty list defaults
            to using all available outputs
        """
        self._distill_output_keys = value

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        distillation_teacher: Module = None,
        **kwargs,
    ):
        """
        Store the teacher model for distillation. Must be provided by the
        distillation_teacher kwarg to the manager/modifier initializer

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param distillation_teacher: teacher module to perform knowledge distillation
            with. Must take inputs of the same shape as the base module and produce
            outputs of the same type and structure
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)

        if distillation_teacher is None:
            raise ValueError(
                "A valid teacher module must be provided to the "
                "distill_teacher kwarg in the modifier/manager initialize(...) function"
            )

        if not isinstance(distillation_teacher, Module):
            raise ValueError(
                "distill_teacher must be a Module. received type "
                f"{type(distillation_teacher)}"
            )

        self._teacher = distillation_teacher
        self._check_distillation_update(module, epoch, steps_per_epoch=0)

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), sets a hook for tracking student module inputs and outputs
        for distillation
        If end_pending(), removes hook for distillation tracking

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_distillation_update(module, epoch, steps_per_epoch)

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        pending = (
            self.start_pending(epoch, steps_per_epoch)
            or self.end_pending(epoch, steps_per_epoch)
            or (not self._distillation_enabled and self._is_distillation_epoch(epoch))
        )

        return pending

    def loss_update(
        self,
        loss: Tensor,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ) -> Tensor:
        """
        Updates the bass loss with the distillation loss

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: loss tensor with knowledge distillation loss added
        """
        loss = super().loss_update(loss, module, optimizer, epoch, steps_per_epoch)

        if not self._distillation_enabled:
            return loss

        if self._student_outputs is None or self._student_inputs is None:
            raise RuntimeError(
                "A forward pass of the module must be run before calling loss_update "
                "with a DistillationModifier"
            )

        # ensure that teacher model is in eval mode and on correct device
        self._teacher.eval()
        target_device = (
            self._student_inputs.device
            if isinstance(self._student_inputs, Tensor)
            else self._student_inputs[0].device
            if isinstance(self._student_inputs, Iterable)
            else [
                tens.device
                for tens in self._student_inputs.values()
                if isinstance(tens, Tensor)
            ][0]
        )
        self._teacher.to(target_device)

        teacher_outputs = tensors_module_forward(
            self._student_inputs, self._teacher, check_feat_lab_inp=False
        )

        assert type(self._student_outputs) == type(
            teacher_outputs
        ), "Student and teacher models must have the same output type"

        distill_losses = []
        if isinstance(self._student_outputs, Tensor):
            distill_losses.append(
                self._calc_distill_loss(self._student_outputs, teacher_outputs)
            )
        elif isinstance(self._student_outputs, Dict):
            for key in self._distill_output_keys or self._student_outputs:
                distill_losses.append(
                    self._calc_distill_loss(
                        self._student_outputs[key], teacher_outputs[key]
                    )
                )
        elif isinstance(self._student_outputs, Iterable):
            for idx in self._distill_output_keys or range(len(self._student_outputs)):
                distill_losses.append(
                    self._calc_distill_loss(
                        self._student_outputs[idx], teacher_outputs[idx]
                    )
                )

        # get distillation loss as average of individual output distillation loss values
        distill_loss = sum(distill_losses) / len(distill_losses)
        return ((1.0 - self._hardness) * loss) + (self._hardness * distill_loss)

    def finalize(
        self, module: Optional[Module] = None, reset_loggers: bool = True, **kwargs
    ):
        """
        Cleans up any state and hooks

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().finalize(module, reset_loggers, **kwargs)
        self._teacher = None
        self._disable_student_hook()

    def _calc_distill_loss(self, student_val: Tensor, teacher_val: Tensor) -> Tensor:
        return (
            TF.kl_div(
                input=TF.log_softmax(student_val / self._temperature, dim=-1),
                target=TF.softmax(teacher_val / self._temperature, dim=-1),
                reduction="batchmean",
            )
            * (self._temperature ** 2)
        )

    def _check_distillation_update(
        self, module: Module, epoch: float, steps_per_epoch: int
    ):
        if self.start_pending(epoch, steps_per_epoch) or (
            not self._distillation_enabled and self._is_distillation_epoch(epoch)
        ):
            self._set_student_hook(module)
            self._distillation_enabled = True

        if self.end_pending(epoch, steps_per_epoch):
            self._disable_student_hook()
            self._distillation_enabled = False

    def _set_student_hook(self, module: Module):
        # delete hook if already exists
        self._disable_student_hook()

        def _track_inputs_and_outputs_hook(mod, inputs, outputs):
            self._student_inputs = inputs
            self._student_outputs = outputs

        self._track_student_hook = module.register_forward_hook(
            _track_inputs_and_outputs_hook
        )

    def _disable_student_hook(self):
        if self._track_student_hook is not None:
            self._track_student_hook.remove()
            self._track_student_hook = None
            self._student_inputs = None
            self._student_outputs = None

    def _is_distillation_epoch(self, epoch):
        return self.start_epoch <= epoch < self.end_epoch
