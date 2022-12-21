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
Modifier for performing knowledge distillation via feature imitation.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from torch.nn import Module

from sparseml.optim import ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.pytorch.utils import BaseLogger


__all__ = [
    "PerLayerDistillationModifier",
]

_LOGGER = logging.getLogger(__name__)

_DISTILLATION_TYPES = [torch.nn.Conv2d, torch.nn.Linear]


@PyTorchModifierYAML()
class PerLayerDistillationModifier(BaseDistillationModifier):
    """
    Adds a knowledge distillation loss based on the feature imitation loss.
    A distillation_teacher module may be provided as a kwarg to
    the Manager initialization and loss_update(loss) must be called before any
    backwards pass in the integrated training flow.
    If no teacher model is provided, then self-distillation will be used.
    The feature difference between teacher and student can be weighted spatially
    by a weighing function.

    # Sample yaml:

    ```yaml
    !PerLayerDistillationModifier
        gain: 2.0
        start_epoch: 0.0
        project_features: True
        student_layer_names
    ```

    # Parameters

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param distill_output_keys: List of keys for the module outputs to use for
        distillation if multiple outputs are present. None or empty list defaults
        to using all available outputs
    :param teacher_input_keys: List of keys to filter the inputs by before
        passing into the teacher. None or empty list defaults to using
        all available inputs
    :param update_frequency:
    :param gain: How much to weight the distillation loss. Default is 1.5
    :param normalize: Whether to normalize the output difference by the
        the magnitude of the teacher's output
    :param epsilon: Small value used to avoid division by zero when normalization
        is used. Default is 1.e-6
    """

    def __init__(
        self,
        gain: float,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        normalize: bool = True,
        student_layer_names: Optional[List[str]] = None,
        teacher_layer_names: Optional[List[str]] = None,
        project_features: bool = False,
        epsilon: float = 1.0e-6,
    ):
        if (
            student_layer_names is not None
            and teacher_layer_names is not None
            and len(student_layer_names) != len(teacher_layer_names)
        ):
            raise ValueError(
                "Student and teacher layer names must have the same number of elements"
            )

        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            distill_output_keys=None,
            teacher_input_keys=None,
            update_frequency=update_frequency,
        )
        self._gain = gain
        self._normalize = normalize
        self._student_layer_names = student_layer_names
        self._teacher_layer_names = teacher_layer_names
        self._project_features = project_features
        self._epsilon = epsilon
        self._cached_student_output = None
        self._cached_teacher_output = None
        self._student_handles = None
        self._teacher_handles = None
        self._projection = None
        self._student_output_shapes = None
        self._teacher_output_shapes = None

    @ModifierProp()
    def gain(self) -> float:
        """
        :return: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        return self._gain

    @gain.setter
    def gain(self, value: float):
        """
        :params value: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        self._gain = value

    @ModifierProp()
    def normalize(self) -> bool:
        """
        :return: whether to normalize distillation loss by magnitude of teacher output
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        """
        :params value: whether to normalize distillation loss
            by magnitude of teacher output
        """
        self._normalize = value

    @ModifierProp()
    def epsilon(self) -> float:
        """
        :return: small value to avoid division per zero when normalization is used
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        """
        :params value: small value to avoid division per zero when normalization is used
        """
        self._epsilon = value

    @ModifierProp()
    def student_layer_names(self) -> List[str]:
        return self._student_layer_names

    @student_layer_names.setter
    def student_layer_names(self, value: List[str]):
        self._student_layer_names = value

    @ModifierProp()
    def teacher_layer_names(self) -> List[str]:
        return self._teacher_layer_names

    @teacher_layer_names.setter
    def teacher_layer_names(self, value: List[str]):
        self._teacher_layer_names = value

    @ModifierProp()
    def project_features(self) -> bool:
        return self._project_features

    @project_features.setter
    def project_features(self, value: bool):
        self._project_features = value

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
        super().initialize(module, epoch, loggers, distillation_teacher, **kwargs)

        if isinstance(distillation_teacher, Module):
            self._teacher = distillation_teacher
        else:
            raise ValueError(
                "unrecognized value for distillation_modifier given of "
                f"{distillation_teacher}. "
                "To disable set to 'disable' and for self attention set to 'self'"
            )

        self._cached_student_output = {}
        self._cached_teacher_output = {}
        self._student_output_shapes = {}
        self._teacher_output_shapes = {}
        self._student_handles = []
        self._teacher_handles = []

        cached_student_layers: Dict[str, torch.nn.Module] = {}
        if self.student_layer_names is None:
            _find_layers_by_type(module, cached_student_layers)
            self.student_layer_names = list(cached_student_layers.keys())
        else:
            _find_layers_by_name(
                module, self.student_layer_names, cached_student_layers
            )
        _LOGGER.info("Distilling student layers: %s", self.student_layer_names)

        cached_teacher_layers: Dict[str, torch.nn.Module] = {}
        if self.teacher_layer_names is None:
            _find_layers_by_type(self._teacher, cached_teacher_layers)
            self.teacher_layer_names = list(cached_teacher_layers.keys())
        else:
            _find_layers_by_name(
                self._teacher, self.teacher_layer_names, cached_teacher_layers
            )
        _LOGGER.info("Distilling teacher layers: %s", self.teacher_layer_names)

        if len(self.teacher_layer_names) != len(self.student_layer_names):
            raise ValueError(
                "Found different numbers of teacher and student layers to distill. "
                "Set teacher_layer_names and student_layer_names explicitly."
            )

        for layer_name in cached_student_layers:
            self._student_handles.append(
                cached_student_layers[layer_name].register_forward_hook(
                    _create_cache_output_hook(
                        layer_name,
                        self._cached_student_output,
                        self._student_output_shapes,
                    )
                )
            )

        for layer_name in cached_teacher_layers:
            self._teacher_handles.append(
                cached_teacher_layers[layer_name].register_forward_hook(
                    _create_cache_output_hook(
                        layer_name,
                        self._cached_teacher_output,
                        self._teacher_output_shapes,
                    )
                )
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
        for handle in self._student_handles:
            handle.remove()
        for handle in self._teacher_handles:
            handle.remove()
        self._student_handles = None
        self._teacher_handles = None
        self._cached_student_output = None
        self._cached_teacher_output = None

    def compute_distillation_loss(self, optimizer, **kwargs):
        distillation_loss = 0.0

        if self.project_features and self._projection is None:
            self._initialize_projection()
            student_module_output = self._cached_student_output[
                self.student_layer_names[0]
            ]
            device = student_module_output.device
            parameters = [p.weight for p in self._projection]
            optimizer.add_param_group({"params": parameters})
            self._projection = [p.to(device) for p in self._projection]

        for index in range(len(self.student_layer_names)):
            student_module_output = self._cached_student_output[
                self.student_layer_names[index]
            ]
            teacher_module_output = self._cached_teacher_output[
                self.teacher_layer_names[index]
            ]

            if self.project_features:
                student_module_output = self._projection[index](
                    student_module_output.float()
                )

            output_difference = torch.mean(
                (student_module_output - teacher_module_output) ** 2,
            )

            if self.normalize:
                teacher_output_magnitude = torch.mean(teacher_module_output ** 2)
                output_difference /= teacher_output_magnitude + self.epsilon

            distillation_loss += output_difference

        return distillation_loss

    def _initialize_projection(self):
        self._projection = []
        for index in range(len(self.student_layer_names)):
            student_shape = self._student_output_shapes[self.student_layer_names[index]]
            teacher_shape = self._teacher_output_shapes[self.teacher_layer_names[index]]
            if len(student_shape) == 4:
                student_features = student_shape[1]
                teacher_features = teacher_shape[1]
                self._projection.append(
                    torch.nn.Conv2d(
                        in_channels=student_features,
                        out_channels=teacher_features,
                        kernel_size=1,
                        bias=False,
                    )
                )
            else:
                student_features = student_shape[-1]
                teacher_features = teacher_shape[-1]
                self._projection.append(
                    torch.nn.Linear(
                        in_features=student_features,
                        out_features=teacher_features,
                        bias=False,
                    )
                )

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss


def _create_cache_output_hook(layer_name, outputs, outputs_shape):
    def forward_hook_fn(layer, inp, out):
        outputs[layer_name] = out
        if layer_name not in outputs_shape:
            outputs_shape[layer_name] = out.shape

    return forward_hook_fn


def _find_layers_by_type(
    layer_module: torch.nn.Module,
    cached_layers: Dict[str, torch.nn.Module],
    name: str = "",
):
    if type(layer_module) in _DISTILLATION_TYPES:
        cached_layers[name] = layer_module
    for layer_module, child in layer_module.named_children():
        _find_layers_by_type(
            child,
            cached_layers,
            name + "." + layer_module if name != "" else layer_module,
        )


def _find_layers_by_name(
    layer_module: torch.nn.Module,
    layer_names: List[str],
    cached_layers: Dict[str, torch.nn.Module],
    name: str = "",
):
    if name in layer_names:
        cached_layers[name] = layer_module
    for layer_module, child in layer_module.named_children():
        _find_layers_by_name(
            child,
            layer_names,
            cached_layers,
            name + "." + layer_module if name != "" else layer_module,
        )
