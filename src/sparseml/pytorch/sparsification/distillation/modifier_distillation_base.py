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
Base modifier for performing model distillation
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    ScheduledModifier,
    ScheduledUpdateModifier,
)
from sparseml.pytorch.utils import BaseLogger, device_of, tensors_module_forward
from sparseml.sparsification import SparsificationTypes


__all__ = ["BaseDistillationModifier", "kl_logsoftmax", "kldiv_loss"]

_LOGGER = logging.getLogger(__name__)


def kl_logsoftmax(
    x: Tensor, y: Tensor, temperature: Union[float, Tensor], dim: int = -1
) -> Tensor:
    number_items = x.numel() / y.size(dim)
    return (
        TF.kl_div(
            input=TF.log_softmax(x / temperature, dim=dim),
            target=TF.log_softmax(y / temperature, dim=dim),
            log_target=True,
            reduction="sum",
        )
        * (temperature ** 2)
        / number_items
    )


def kldiv_loss(
    student_outputs,
    teacher_outputs,
    temperature,
    output_keys=None,
    dim=-1,
):
    distill_head_output_losses = []
    if isinstance(student_outputs, Tensor):
        distill_head_output_losses.append(
            kl_logsoftmax(student_outputs, teacher_outputs, temperature, dim)
        )
    elif isinstance(student_outputs, Mapping):
        for key in output_keys or student_outputs:
            distill_head_output_losses.append(
                kl_logsoftmax(
                    student_outputs[key], teacher_outputs[key], temperature, dim
                )
            )
    elif isinstance(student_outputs, Iterable):
        for idx in output_keys or range(len(student_outputs)):
            distill_head_output_losses.append(
                kl_logsoftmax(
                    student_outputs[idx], teacher_outputs[idx], temperature, dim
                )
            )
    kldiv_output_loss = (
        sum(distill_head_output_losses) / len(distill_head_output_losses)
        if distill_head_output_losses
        else 0.0
    )
    return kldiv_output_loss


class BaseDistillationModifier(ScheduledUpdateModifier):
    """
    Adds a knowledge distillation loss based on a teacher model during the
    loss_update phase of the SparseML lifecycle. A distillation_teacher
    module may be provided as a kwarg to the Manager initialization and
    loss_update(loss) must be called before any backwards pass in the integrated
    training flow. If no teacher model is provided, then self distillation
    will be used.
    Specific instances of knowledge distillation can be defined by
    implementing methods to compute the distillation loss and to combine
    the distillation loss with other loss terms to compute the total loss.

    Common parameters across different distillation implementations:

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param distill_output_keys: list of keys for the module outputs to use for
        distillation if multiple outputs are present. None or empty list defaults
        to using all available outputs
    :param teacher_input_keys: list of keys to filter the inputs by before
        passing into the teacher. None or empty list defaults to using
        all available inputs
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        distill_output_keys: List[Any] = None,
        teacher_input_keys: List[Any] = None,
        update_frequency: float = -1.0,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )
        self._distill_output_keys = distill_output_keys
        self._teacher_input_keys = teacher_input_keys

        self._teacher = None
        self._distillation_enabled = False

        self._logged_loss_terms = {}

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.distillation]

    @ModifierProp()
    def distill_output_keys(self) -> Optional[List[Any]]:
        """
        :return: list of keys for the module outputs to use for
            distillation if multiple outputs are present. None or empty list defaults
            to using all available outputs
        """
        return self._distill_output_keys

    @distill_output_keys.setter
    def distill_output_keys(self, value: Optional[List[Any]]):
        """
        :params value: list of keys for the module outputs to use for
            distillation if multiple outputs are present. None or empty list defaults
            to using all available outputs
        """
        self._distill_output_keys = value

    @ModifierProp()
    def teacher_input_keys(self) -> Optional[List[Any]]:
        """
        :return: list of keys to filter the inputs by before
            passing into the teacher. None or empty list defaults to using
            all available inputs
        """
        return self._teacher_input_keys

    @teacher_input_keys.setter
    def teacher_input_keys(self, value: Optional[List[Any]]):
        """
        :params value: list of keys to filter the inputs by before
            passing into the teacher. None or empty list defaults to using
            all available inputs
        """
        self._teacher_input_keys = value

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        distillation_teacher: Union[Module, str] = "disable",
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

        if distillation_teacher == "disable":
            _LOGGER.warning(
                "distillation_teacher set to disable, disabling distillation modifier"
            )
            self._distillation_enabled = False
        elif distillation_teacher == "self":
            self._distillation_enabled = True
            _LOGGER.info(
                "distillation_teacher set to self attention, "
                "instantiating self distillation at start_epoch"
            )
        elif callable(distillation_teacher):
            self._teacher = distillation_teacher
            self._distillation_enabled = True
            _LOGGER.info("distillation modifier using distillation_teacher object")
        else:
            raise ValueError(
                "unrecognized value for distillation_modifier given of "
                f"{distillation_teacher}. "
                "To disable set to 'disable' and for self attention set to 'self'"
            )

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        return self._distillation_enabled and super().update_ready(
            epoch, steps_per_epoch
        )

    @ScheduledModifier.log_call
    def loss_update(
        self,
        loss: Tensor,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
        student_outputs: Union[Tensor, Dict, Iterable] = None,
        student_inputs: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]] = None,
        teacher_inputs: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]] = None,
        student_labels: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]] = None,
        teacher_labels: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Updates the loss with the distillation loss

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :param student_outputs: predictions resulting from evaluating student
            model on student inputs
        :param student_inputs: inputs for student model
        :param teacher_inputs: inputs for teacher model. If None, will use
            same inputs as student
        :param student_labels: labels associated with student input data
        :param teacher_labels: labels associated with teacher input data.
            If None, will use student labels.
        :return: loss tensor with knowledge distillation loss added
        """
        loss = super().loss_update(
            loss, module, optimizer, epoch, steps_per_epoch, **kwargs
        )
        self._logged_loss_terms["task_loss"] = loss

        if not self.update_ready(epoch, steps_per_epoch):
            return loss

        if student_outputs is None or student_inputs is None:
            raise ValueError(
                "Student outputs and student inputs are required for "
                "distillation loss update"
            )

        if teacher_inputs is None:
            teacher_inputs = (
                student_inputs
                if not self._teacher_input_keys
                else {key: student_inputs[key] for key in self._teacher_input_keys}
            )

        if teacher_labels is None:
            teacher_labels = student_labels

        # copy to keep from updating student's inputs
        teacher_inputs = deepcopy(teacher_inputs)

        if self._teacher == "self":
            _LOGGER.info("Copying current models state for self distillation")
            self._teacher = deepcopy(module)

        # ensure that teacher model is in eval mode and on correct device
        self._teacher.eval()
        teacher_device = next(self._teacher.parameters()).device
        inputs_device = device_of(teacher_inputs)

        if teacher_device != inputs_device:
            _LOGGER.info(
                f"Teacher device {teacher_device} does not match "
                f"inputs device {inputs_device}, moving teacher to correct device"
            )
            self._teacher.to(inputs_device)

        with torch.no_grad():
            teacher_outputs = tensors_module_forward(
                teacher_inputs, self._teacher, check_feat_lab_inp=False
            )

        if type(student_outputs) != type(teacher_outputs):
            raise ValueError(
                f"Student output type of {type(student_outputs)} must match "
                f"teacher output type of {type(teacher_outputs)}"
            )

        distillation_loss = self.compute_distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            student_labels=student_labels,
            teacher_labels=teacher_labels,
        )

        total_loss = self.compute_total_loss(loss, distillation_loss)
        self._logged_loss_terms.update(
            {"distillation_loss": distillation_loss, "total_loss": total_loss}
        )

        return total_loss

    def log_update(
        self,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ):
        """
        log the latest set of losses

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        self.log_named_scalars(
            name_value_pairs=self._logged_loss_terms.items(),
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
        )

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
        self._distillation_enabled = False

    def compute_distillation_loss(self, student_outputs, teacher_outputs, labels):
        raise NotImplementedError()

    def compute_total_loss(self, loss, distillation_loss):
        raise NotImplementedError()
