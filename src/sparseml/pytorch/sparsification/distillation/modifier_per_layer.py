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
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from sparseml.optim import ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.pytorch.utils import BaseLogger


__all__ = [
    "PerLayerDistillationModifier",
    "DISTILL_PARAM_GROUP_KEY",
]

DISTILL_PARAM_GROUP_KEY = "distillation_projection_params"

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

    The connection between layers in the student model and layers in the teacher
    model is controlled via the `teacher_layer_names` and `student_layer_names`
    fields. **These fields must be the same length!** The i'th item in both
    of these arrays are paired together.

    # Sample yaml:

    ```yaml
    !PerLayerDistillationModifier
        gain: 2.0
        start_epoch: 0.0
        end_epoch: 1.0
        update_frequency: 0.2
        normalize: True
        student_layer_names: ["layer2.0.conv2"]
        teacher_layer_names: ["layer2.0.conv2"]
        project_features: True
        epsilon: 1e-6
    ```

    # Parameters

    :param gain: How much to weight the distillation loss.
        Default is `1.5`
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to
            update at between start and end
    :param normalize: Whether to normalize the output difference by the
        the magnitude of the teacher's output.
        Default is `True`.
    :param student_layer_names: List of layer names to distill.
        *Must be same length as teacher_layer_names.*
    :param teacher_layer_names: List of layer names to distill from.
        *Must be same length as student_layer_names.*
    :param project_features: Whether to project the output of student layers to the
        same size as the output of the teacher layers.
        Default is `True`
    :param epsilon: Small value used to avoid division by zero when normalization
        is used. Default is `1e-6`
    """

    def __init__(
        self,
        gain: float = 1.5,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        update_frequency: float = -1.0,
        normalize: bool = True,
        student_layer_names: Optional[List[str]] = None,
        teacher_layer_names: Optional[List[str]] = None,
        project_features: bool = True,
        epsilon: float = 1e-6,
    ):
        if (
            student_layer_names is not None
            and teacher_layer_names is not None
            and len(student_layer_names) != len(teacher_layer_names)
        ):
            raise ValueError(
                "Student and teacher layer names must have the same number of elements"
            )

        if student_layer_names is None and teacher_layer_names is not None:
            _LOGGER.info(
                "Distilling same layer names for teacher and student: %s",
                teacher_layer_names,
            )
            student_layer_names = teacher_layer_names.copy()

        if teacher_layer_names is None and student_layer_names is not None:
            _LOGGER.info(
                "Distilling same layer names for teacher and student: %s",
                student_layer_names,
            )
            teacher_layer_names = student_layer_names.copy()

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

        self._cached_student_output: Dict[str, torch.Tensor] = {}
        self._cached_teacher_output: Dict[str, torch.Tensor] = {}
        self._student_handles: List[RemovableHandle] = []
        self._teacher_handles: List[RemovableHandle] = []
        self._projection: Dict[str, torch.nn.Module] = {}
        self._student_output_shapes: Dict[str, torch.Size] = {}
        self._teacher_output_shapes: Dict[str, torch.Size] = {}
        self._loaded_projection = None

    def _reset_cache(self):
        self._cached_student_output.clear()
        self._cached_teacher_output.clear()
        self._student_handles.clear()
        self._teacher_handles.clear()
        self._projection.clear()
        self._student_output_shapes.clear()
        self._teacher_output_shapes.clear()

    @ModifierProp()
    def gain(self) -> float:
        """
        :return: how much to weight the distillation loss vs the base loss
            (e.g. gain of 0.6 will return 0.6 * distill_loss + base_loss)
        """
        return self._gain

    @gain.setter
    def gain(self, value: float):
        """
        :params value: how much to weight the distillation loss vs the base loss
            (e.g. gain of 0.6 will return 0.6 * distill_loss + base_loss)
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
        """
        :return: List of layer names to distill.
        """
        return self._student_layer_names

    @student_layer_names.setter
    def student_layer_names(self, value: List[str]):
        """
        :param value: List of layer names to distill.
            *Must be same length as teacher_layer_names*
        """
        self._student_layer_names = value

    @ModifierProp()
    def teacher_layer_names(self) -> List[str]:
        """
        :return: List of layer names to distill
        """
        return self._teacher_layer_names

    @teacher_layer_names.setter
    def teacher_layer_names(self, value: List[str]):
        """
        :param value: List of layer names to distill.
            *Must be same length as student_layer_names*
        """
        self._teacher_layer_names = value

    @ModifierProp()
    def project_features(self) -> bool:
        """
        :return: Whether to perform projection
            between student and teacher layers
        """
        return self._project_features

    @project_features.setter
    def project_features(self, value: bool):
        """
        :param value: Whether to perform projection
            between student and teacher layers
        """
        self._project_features = value

    def state_dict(self) -> Dict[str, Dict]:
        state = super().state_dict()
        if self.project_features:
            state[DISTILL_PARAM_GROUP_KEY] = {
                name: p.state_dict() for name, p in self._projection.items()
            }
        return state

    def load_state_dict(self, state_dict: Dict[str, Dict], strict: bool = True):
        if self.project_features:
            # save until self._projection is actually initialized after forward
            self._loaded_projection = state_dict.pop(DISTILL_PARAM_GROUP_KEY)
        return super().load_state_dict(state_dict, strict)

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
            with. Pass `"self"` to use self distillation with a teacher
            from a copy of the given module at the start epoch.
            Pass `"disable"` to disable this modifier.
            Defaults to `"disable"`
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, distillation_teacher, **kwargs)

        if not self._distillation_enabled:
            return

        if distillation_teacher == "self":
            self._teacher = deepcopy(module)

        self._reset_cache()

        cached_student_layers: Dict[str, torch.nn.Module] = {}
        if self._student_layer_names is None:
            _update_layers_by_type(module, cached_student_layers)
            self._student_layer_names = list(cached_student_layers.keys())
        else:
            _update_layers_by_name(
                module, self._student_layer_names, cached_student_layers
            )
        _LOGGER.info("Distilling student layers: %s", self._student_layer_names)

        cached_teacher_layers: Dict[str, torch.nn.Module] = {}
        if self._teacher_layer_names is None:
            _update_layers_by_type(self._teacher, cached_teacher_layers)
            self._teacher_layer_names = list(cached_teacher_layers.keys())
        else:
            _update_layers_by_name(
                self._teacher, self._teacher_layer_names, cached_teacher_layers
            )
        _LOGGER.info("Distilling teacher layers: %s", self._teacher_layer_names)

        if len(self._teacher_layer_names) != len(self._student_layer_names):
            raise ValueError(
                "Found different numbers of teacher and student layers to distill. "
                "Set teacher_layer_names and student_layer_names explicitly."
            )

        self._student_handles = [
            layer.register_forward_hook(
                _create_cache_output_hook(
                    name, self._cached_student_output, self._student_output_shapes
                )
            )
            for name, layer in cached_student_layers.items()
        ]

        self._teacher_handles = [
            layer.register_forward_hook(
                _create_cache_output_hook(
                    name, self._cached_teacher_output, self._teacher_output_shapes
                )
            )
            for name, layer in cached_teacher_layers.items()
        ]

    def update(
        self,
        module: Module,
        optimizer: torch.optim.Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ):
        super().update(module, optimizer, epoch, steps_per_epoch)
        if 0 < self.end_epoch <= epoch:
            for handle in self._student_handles:
                handle.remove()
            for handle in self._teacher_handles:
                handle.remove()
            self._reset_cache()

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss

    def compute_distillation_loss(self, optimizer: torch.optim.Optimizer, **kwargs):
        if self.project_features and len(self._projection) == 0:
            # NOTE: have to call initialize here because we need the cached output
            # from the module. i.e. we need forward to have been called already
            self._projection = self._initialize_projection()

            if self._loaded_projection is not None:
                for name, layer in self._projection.items():
                    layer.load_state_dict(self._loaded_projection.pop(name))
                self._loaded_projection = None

        self._patch_state_dict_loading(optimizer)

        distillation_loss = 0.0
        for student_name, teacher_name in zip(
            self._student_layer_names, self._teacher_layer_names
        ):
            student_output = self._cached_student_output[student_name]
            if self.project_features:
                student_output = self._projection[student_name](student_output.float())

            teacher_output = self._cached_teacher_output[teacher_name]

            output_mse = (student_output - teacher_output).square().mean()
            if self.normalize:
                teacher_output_magnitude = teacher_output.square().mean()
                output_mse /= teacher_output_magnitude + self.epsilon

            distillation_loss += output_mse

        return distillation_loss

    def _patch_state_dict_loading(self, optimizer):
        if _get_projection_param_group_idx(optimizer.param_groups) is None:
            optimizer.add_param_group(
                {
                    DISTILL_PARAM_GROUP_KEY: True,
                    "params": [p.weight for p in self._projection.values()],
                }
            )

            _old_state_dict_fn = optimizer.state_dict
            _old_load_state_dict_fn = optimizer.load_state_dict

            def state_dict_without_projection():
                # Remove the param_group we added, as we save that in the modifier's
                # state_dict.
                state = _old_state_dict_fn()
                state = deepcopy(state)
                idx = _get_projection_param_group_idx(state["param_groups"])
                assert idx is not None, "optimizer must have had the param_group added"
                state["param_groups"].pop(idx)
                return state

            def load_state_dict_without_projection(state_dict):
                # NOTE: state_dict won't have the state for the projections
                # because we removed that in `state_dict_without_projection`.
                # but `optimizer` will have an additional param_group, since
                # it was added above.
                # so we:
                # 1. remove the param group from optimizer
                # 2. call load_state_dict
                # 3. re-add the param group to the optimizer
                idx = _get_projection_param_group_idx(optimizer.param_groups)
                assert idx is not None, "optimizer must have had the param_group added"
                param_group = optimizer.param_groups.pop(idx)
                _old_load_state_dict_fn(state_dict)
                optimizer.add_param_group(param_group)

            optimizer.state_dict = state_dict_without_projection
            optimizer.load_state_dict = load_state_dict_without_projection

    def _initialize_projection(self) -> Dict[str, torch.Tensor]:
        device = self._cached_student_output[self.student_layer_names[0]].device

        assert len(self._student_output_shapes) == len(self._student_layer_names)
        assert len(self._teacher_output_shapes) == len(self._teacher_layer_names)

        projections = {}
        for student_name, teacher_name in zip(
            self._student_layer_names, self._teacher_layer_names
        ):
            student_shape = self._student_output_shapes[student_name]
            teacher_shape = self._teacher_output_shapes[teacher_name]
            if len(student_shape) == 4:
                projections[student_name] = torch.nn.Conv2d(
                    in_channels=student_shape[1],
                    out_channels=teacher_shape[1],
                    kernel_size=1,
                    bias=False,
                ).to(device)
            else:
                projections[student_name] = torch.nn.Linear(
                    in_features=student_shape[-1],
                    out_features=teacher_shape[-1],
                    bias=False,
                ).to(device)

        return projections


def _create_cache_output_hook(
    layer_name: str,
    outputs: Dict[str, torch.Tensor],
    outputs_shape: Dict[str, torch.Size],
):
    def forward_hook_fn(layer, inp, out):
        outputs[layer_name] = out
        if layer_name not in outputs_shape:
            outputs_shape[layer_name] = out.shape

    return forward_hook_fn


def _update_layers_by_type(
    layer_module: torch.nn.Module,
    cached_layers: Dict[str, torch.nn.Module],
    name: str = "",
):
    if type(layer_module) in _DISTILLATION_TYPES:
        cached_layers[name] = layer_module
    for layer_module, child in layer_module.named_children():
        _update_layers_by_type(
            child,
            cached_layers,
            name + "." + layer_module if name != "" else layer_module,
        )


def _update_layers_by_name(
    layer_module: torch.nn.Module,
    layer_names: List[str],
    cached_layers: Dict[str, torch.nn.Module],
    name: str = "",
):
    if name in layer_names:
        cached_layers[name] = layer_module
    for layer_module, child in layer_module.named_children():
        _update_layers_by_name(
            child,
            layer_names,
            cached_layers,
            name + "." + layer_module if name != "" else layer_module,
        )


def _get_projection_param_group_idx(param_groups: List[Dict]) -> Optional[int]:
    """
    :return: Optional index where the param group is if it is found.
    """
    for idx, group in enumerate(param_groups):
        if DISTILL_PARAM_GROUP_KEY in group:
            return idx
