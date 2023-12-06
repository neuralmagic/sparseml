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

from typing import Any, Dict

import torch
from torch.nn import Module

from sparseml.core import Event, EventType, State
from sparseml.modifiers.distillation.output.base import OutputDistillationModifier
from sparseml.modifiers.distillation.utils.pytorch import KDFactory, KDModuleWrapper


__all__ = ["OutputDistillationModifierPyTorch"]


class OutputDistillationModifierPyTorch(OutputDistillationModifier):
    wrappers_: Dict[str, Any] = None
    fsdp_active_: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        if (
            state.framework is None
            or state.model is None
            or state.teacher_model is None
        ):
            return False

        self.wrappers_ = {}
        if kwargs.get("fsdp_active"):
            self.fsdp_active_ = True

        for target in (
            self.targets if isinstance(self.targets, list) else [self.targets]
        ):
            if isinstance(target, tuple):
                model_target, teacher_target = target
            else:
                model_target, teacher_target = target, target

            model_layers = state.model.get_layers(model_target)
            teacher_layers = state.teacher_model.get_layers(teacher_target)

            if len(model_layers) < 1:
                raise ValueError(f"no model layers found for target {target}")

            if len(model_layers) != len(teacher_layers):
                raise ValueError(
                    f"model and teacher model layers for target {target} do not match"
                )

            # Ensure the student and teacher layer are on the same device
            device = None
            for (key, student_layer), teacher_layer in zip(
                model_layers.items(), teacher_layers.values()
            ):
                for name, param in student_layer.named_parameters():
                    device = param.device
                    teacher_param = getattr(teacher_layer, name)
                    teacher_param.data = teacher_param.to(device)

                for name, buffer in student_layer.named_buffers():
                    if hasattr(teacher_layer, name):
                        device = buffer.device
                        teacher_buffer = getattr(teacher_layer, name)
                        teacher_buffer.data = teacher_buffer.to(device)

                wrapper = self._create_wrapper(student_layer, teacher_layer, state)
                kd_comparison_buffer = getattr(wrapper, wrapper.KD_COMPARISON_BUFFER)
                kd_comparison_buffer.data = kd_comparison_buffer.to(device)
                state.model.set_layer(key, wrapper)
                self.wrappers_[key] = wrapper
                wrapper.kd_enabled = True

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        for key, wrapper in self.wrappers_.items():
            state.model.set_layer(key, wrapper.student_layer)
            del wrapper

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        for wrapper in self.wrappers_.values():
            wrapper.kdenabled = True

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.LOSS_CALCULATED and event.should_update(
            self.start, self.end, self.update
        ):
            comparisons = [
                wrapper.kd_last_comparison for wrapper in self.wrappers_.values()
            ]
            state.loss = (
                self.orig_scale * kwargs["loss"]  # model output loss
                + self.distill_scale * torch.stack(comparisons).mean()  # distill loss
            )

    def on_end(self, state: State, event: Event, **kwargs):
        for wrapper in self.wrappers_.values():
            wrapper.kdenabled = False

    def _create_wrapper(
        self, student_layer: Module, teacher_layer: Module, state: State
    ) -> KDModuleWrapper:
        projections = (
            KDFactory.create_projection(
                self.projection,
                student_layer,
                teacher_layer,
                state,
                **(self.projection_args or {}),
            )
            if self.projection
            else None
        )
        comparison = KDFactory.create_comparison(
            self.comparison,
            student_layer,
            teacher_layer,
            state,
            **(self.comparison_args or {}),
        )

        transforms = []
        if self.transforms:
            tmp_transforms = (
                self.transforms
                if isinstance(self.transforms, list)
                else [self.transforms]
            )
            tmp_transform_args = [
                args
                for args in (
                    self.transforms_args
                    if isinstance(self.transforms_args, list)
                    else [self.transforms_args if self.transforms_args else {}]
                )
                for _ in range(len(tmp_transforms))
            ]

            for transform, transform_args in zip(tmp_transforms, tmp_transform_args):
                transforms.append(
                    KDFactory.create_transform(
                        transform,
                        student_layer,
                        teacher_layer,
                        state,
                        **transform_args,
                    )
                )

        return KDModuleWrapper(
            student_layer=student_layer,
            teacher_layer=teacher_layer,
            projections=projections,
            transforms=transforms,
            comparison=comparison,
            fsdp_active=self.fsdp_active_,
        )
