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
    |       topk: 5
    |       multigpu_distillation: True
    |       loss: kl-divergence

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
        multigpu_distillation: bool = False,
        loss: str = 'kl-divergence',
        topk: int = -1,

    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )
        self._hardness = hardness
        self._temperature = temperature
        self._distill_output_keys = distill_output_keys or []
        self._loss = loss
        self._topk = topk
        self._multigpu_distillation = multigpu_distillation


        self._teacher = None
        self._distillation_enabled = False
        self._track_student_hook = None
        self._student_inputs = None  # last forward inputs to student module
        self._student_outputs = None  # last forward outputs of student module
        self._disable_distillation = False

    @ModifierProp()
    def loss(self) -> str:
        """
        :return: the name of the loss function for distillation
            (e.g. using the kl-divergence will return the kl-divergence)
        """
        return self._loss
    
    @loss.setter
    def loss(self, value: str):
        """
        :params value: what loss function to use for distillation
            (e.g. loss of kl-divergence will return kl-divergence)
        """
        self._loss = value

    @ModifierProp()
    def topk(self) -> int:
        """
        :return: how many topk values to take from teacher logits for distillation
            (e.g. topk of 2 will return 2 hardness used to keep only top 2 values from logits vals, idx = teacher_val.topk(self._topk)
        """
        return self._topk
    
    @topk.setter
    def topk(self, value: int):
        """
        :params value: how many values from teacher logits to use for distillation
            (e.g. topk of 2 will return 2 hardness used to keep only top 2 values from logits vals, idx = teacher_val.topk(self._topk)
        """

        self._topk = value

    @ModifierProp()
    def multigpu_distillation (self) -> bool:
        """
        :return: bool on perfoming distillation across gpus
            (e.g. multigpu_distillation of true will create a teacher on each gpu)
        """
        return self._multigpu_distillation
    
    

    @multigpu_distillation.setter
    def multigpu_distillation(self, value: int):
        """
        :params value: bool on perfoming distillation across gpus
            (e.g. multigpu_distillation of true will create a teacher on each gpu)
        """
        
        self._multigpu_distillation = value

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
            if self._multigpu_distillation:
                _LOGGER.info(
                    "Setting up per device distillation"
                )
                self.num_gpus = torch.cuda.device_count()
                self._teacher = [self._teacher for i in range(self.num_gpus)]
                for i in range(0,self.num_gpus):
                    _LOGGER.info(
                        "Moving teacher to GPU %i ", i
                    )
                    self._teacher[i] = self._teacher[i].to(i)

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
        if self._multigpu_distillation:
            # If we are using distillation with multiple GPUs we run the teacher model on each device with respective student output. 
            # This allows us to maximize batch size as without that the teacher forward pass would be of size batch_size * num_gpus which can overload GPU memory
            input_ids = torch.split(teacher_inputs['input_ids'], int(teacher_inputs['input_ids'].shape[0]/self.num_gpus))
            token_type_ids = torch.split(teacher_inputs['token_type_ids'], int(teacher_inputs['token_type_ids'].shape[0]/self.num_gpus))
            attention_mask = torch.split(teacher_inputs['attention_mask'], int(teacher_inputs['attention_mask'].shape[0]/self.num_gpus))
            if isinstance(student_outputs, Tensor):
                student_outputs = torch.split(student_outputs, int(student_outputs.shape[0]/self.num_gpus))
            elif isinstance(student_outputs, Dict):
                split_student_outputs = {}
                for key in self._distill_output_keys or student_outputs:
                    split_student_outputs[key] = torch.split(student_outputs[key], int(student_outputs[key].shape[0]/self.num_gpus))
                student_outputs = split_student_outputs
            elif isinstance(student_outputs, Iterable):
                split_student_outputs = []
                for idx in self._distill_output_keys or range(len(student_outputs)):
                    split_student_outputs = torch.split(student_outputs[idx], int(student_outputs[idx].shape[0]/self.num_gpus))
                student_outputs = split_student_outputs
            
            distill_losses = []
            for i in range(0,self.num_gpus):
                self._teacher[i].eval()
                input_device = self._teacher[i].device
                with torch.no_grad():
                    teacher_outputs = self._teacher[i](
                        input_ids=input_ids[i].to(input_device),
                        token_type_ids=token_type_ids[i].to(input_device),
                        attention_mask=attention_mask[i].to(input_device)
                    )
                
                if isinstance(student_outputs, Tensor):
                    distill_losses.append(
                        self._calc_distill_loss(student_outputs[i].to(input_device), teacher_outputs)
                    )
                elif isinstance(student_outputs, Dict):
                    for key in self._distill_output_keys or student_outputs[i]:
                        distill_losses.append(
                            self._calc_distill_loss(student_outputs[key][i].to(input_device), teacher_outputs[key])
                        )
                elif isinstance(student_outputs, Iterable):
                    for idx in self._distill_output_keys or range(len(student_outputs[i])):
                        distill_losses.append(
                            self._calc_distill_loss(student_outputs[idx][i], teacher_outputs[idx])
                        )
                del teacher_outputs
                        
        else:    
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
        teacher_loss = teacher_loss.to(loss.device)
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
        loss_input = TF.log_softmax(student_val / self._temperature, dim=-1)
        if self._topk > 0:
            vals, idx = teacher_val.topk(self._topk)
            teacher_val = torch.zeros_like(teacher_val)
            teacher_val.view(-1)[idx] = vals
            del vals, idx
        loss_target = TF.softmax(teacher_val / self._temperature, dim=-1)
        if self._loss == 'kl-divergence':
            return (
                TF.kl_div(
                    input=loss_input,
                    target=loss_target,
                    reduction="batchmean",
                )
                * (self._temperature ** 2)
            )
        elif self._loss == 'mean-squared-error':
            loss = torch.nn.MSELoss()
            return (
                loss(loss_input,loss_target) * (self._temperature ** 2)
            )
        else: 
            raise ValueError(
                "Invalid Knowledge distillation method selected. Supported methods are kl-divergence and mean-squared-error"
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
