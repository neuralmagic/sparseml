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


import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from sparseml.optim import ModifierProp
from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.utils import BaseLogger, device_of, tensors_module_forward


__all__ = [
    "DistillationModifier",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class DistillationModifier(ScheduledModifier):
    """
    Adds a knowledge distillation loss based on a teacher model during the
    loss_update phase of the SparseML lifecycle. A distillation_teacher
    module may be provided as a kwarg to the Manager initialization and
    loss_update(loss) must be called before any backwards pass in the integrated
    training flow. If no teacher model is provided, then self distillation
    will be used

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
        self._disable_distillation = False

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
        Store the teacher model for distillation if provided

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param distillation_teacher: teacher module to perform knowledge distillation
            with. If not provided, self distillation will be used with a teacher
             from a copy of the given module at the start epoch. If given string
             "disable" this modifier will not apply distillation of any kind,
             even in the active epoch range
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)

        self._disable_distillation = distillation_teacher == "disable"
        if distillation_teacher is not None:
            _LOGGER.info(
                "Setting teacher module for distillation to distillation_teacher object"
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

        if not self._enabled or self._disable_distillation:
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
        student_outputs: Union[Tensor, Dict, Iterable] = None,
        teacher_inputs: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]] = None,
        **kwargs,
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
        loss = super().loss_update(
            loss, module, optimizer, epoch, steps_per_epoch, **kwargs
        )

        if not self._distillation_enabled or self._disable_distillation:
            return loss

        if student_outputs is None or teacher_inputs is None:
            raise ValueError(
                "Student outputs and teacher inputs are required for "
                "distillation loss update"
            )

        # ensure that teacher model is in eval mode and on correct device
        self._teacher.eval()
        target_device = device_of(teacher_inputs)
        self._teacher.to(target_device)
        with torch.no_grad():
            teacher_outputs = tensors_module_forward(
                teacher_inputs, self._teacher, check_feat_lab_inp=False
            )

        if type(student_outputs) != type(teacher_outputs):
            raise ValueError(
                "Student and teacher models must have the same output type"
            )

        distill_losses = []
        if isinstance(student_outputs, Tensor):
            distill_losses.append(
                self._calc_distill_loss(student_outputs, teacher_outputs)
            )
        elif isinstance(student_outputs, Dict):
            for key in self._distill_output_keys or student_outputs:
                distill_losses.append(
                    self._calc_distill_loss(student_outputs[key], teacher_outputs[key])
                )
        elif isinstance(student_outputs, Iterable):
            for idx in self._distill_output_keys or range(len(student_outputs)):
                distill_losses.append(
                    self._calc_distill_loss(student_outputs[idx], teacher_outputs[idx])
                )

        # get distillation loss as average of individual output distillation loss values
        teacher_loss = sum(distill_losses) / len(distill_losses)
        distillation_loss = ((1.0 - self._hardness) * loss) + (
            self._hardness * teacher_loss
        )
        global_step = kwargs.get("global_step")
        global_step = epoch * steps_per_epoch if global_step is None else global_step
        _log_losses(self.loggers, global_step, loss, teacher_loss, distillation_loss)
        return distillation_loss

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
        if self._disable_distillation:
            _LOGGER.info("Distillation disabled, using default loss")
            return
        if self.start_pending(epoch, steps_per_epoch) or (
            not self._distillation_enabled and self._is_distillation_epoch(epoch)
        ):
            if self._teacher is None:
                _LOGGER.info(
                    "Using self distillation with copy of the module's current state"
                )
                self._teacher = deepcopy(module)
            self._distillation_enabled = True

        if self.end_pending(epoch, steps_per_epoch):
            self._distillation_enabled = False

    def _is_distillation_epoch(self, epoch):
        return self.start_epoch <= epoch < self.end_epoch


def _log_losses(
    loggers: List[BaseLogger],
    global_step: int,
    original_loss: float,
    teacher_loss: float,
    distillation_loss: float,
):
    losses = {
        "original_loss": original_loss,
        "teacher_loss": teacher_loss,
        "distillation_loss": distillation_loss,
    }

    for logger in loggers:
        for (name, loss) in losses.items():
            logger.log_scalar(f"DistillationModifier/{name}", loss.item(), global_step)
