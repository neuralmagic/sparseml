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

from typing import Dict

import torch
from torch.nn import Module

from sparseml.core import Event, EventType, State
from sparseml.modifiers.distillation.output.base import OutputDistillationModifier
from sparseml.modifiers.distillation.utils.pytorch import KDFactory, KDModuleWrapper


__all__ = ["OutputDistillationModifierPyTorch"]


class OutputDistillationModifierPyTorch(OutputDistillationModifier):
    _wrappers: Dict[str, KDModuleWrapper] = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        if (
            state.framework is None
            or state.model is None
            or state.teacher_model is None
        ):
            return False

        self._wrappers = {}

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

            for (key, student_layer), teacher_layer in zip(
                model_layers.items(), teacher_layers.values()
            ):
                wrapper = self._create_wrapper(student_layer, teacher_layer, state)
                state.model.set_layer(key, wrapper)
                self._wrappers[key] = wrapper

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        for key, wrapper in self._wrappers.items():
            state.model.set_layer(key, wrapper.student_layer)
            del wrapper

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        for wrapper in self._wrappers.values():
            wrapper.kdenabled_ = True

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.LOSS_CALCULATED and event.should_update(
            self.start, self.end, self.update
        ):
            comparisons = [
                wrapper.kd_last_comparison for wrapper in self._wrappers.values()
            ]
            state.loss = state.loss + torch.Stack(comparisons).mean()

    def on_end(self, state: State, event: Event, **kwargs):
        for wrapper in self._wrappers.values():
            wrapper.kdenabled_ = False

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
        )
