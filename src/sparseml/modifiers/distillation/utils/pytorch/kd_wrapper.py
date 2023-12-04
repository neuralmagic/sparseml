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

from typing import List, Optional, Sequence, Set, Tuple

import torch
from torch.nn import Module

from sparseml.modifiers.distillation.utils.pytorch.kd_factory import (
    ComparisonFuncType,
    ProjectionFuncType,
    TransformFuncType,
    recursive_apply,
)


__all__ = ["KDModuleWrapper"]


class KDModuleWrapper(Module):
    KD_COMPARISON_BUFFER = "kd_last_comparison"

    def __init__(
        self,
        student_layer: Module,
        teacher_layer: Module,
        projections: Optional[Tuple[ProjectionFuncType, ProjectionFuncType]],
        transforms: Optional[List[TransformFuncType]],
        comparison: ComparisonFuncType,
    ):
        super(KDModuleWrapper, self).__init__()

        self.student_layer = student_layer
        self.teacher_layer = teacher_layer
        self.student_projections = projections[0] if projections is not None else None
        self.teacher_projections = projections[1] if projections is not None else None
        self.kd_transforms = transforms
        self.kd_comparison = comparison
        self.kd_enabled = False
        self.register_buffer(self.KD_COMPARISON_BUFFER, torch.zeros(1))
        self._init_called = True  # make sure this is last property to be set

    # def __getattr__(self, name):
    #     if name.startswith("kd_"):
    #         return getattr(self, name)
    #
    #     return getattr(self.student_layer, name)
    #
    # def __setattr__(self, name, value):
    #     if "_init_called" not in self.__dict__:
    #         super().__setattr__(name, value)
    #     elif name.startswith("kd_"):
    #         super().__setattr__(name, value)
    #     elif hasattr(self.student_layer, name):
    #         setattr(self.student_layer, name, value)
    #     else:
    #         super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        if not self.kd_enabled:
            return self.student_layer(*args, **kwargs)

        org_output = self.student_layer(*args, **kwargs)
        student_output = org_output

        with torch.no_grad():
            teacher_output = self.teacher_layer(*args, **kwargs)

        if self.student_projections is not None:
            for projection in self.student_projections:
                student_output = projection(student_output, teacher_output)

        if self.teacher_projections is not None:
            for projection in self.teacher_projections:
                teacher_output = projection(teacher_output, student_output)

        if self.kd_transforms is not None:
            for transform in self.kd_transforms:
                student_output = transform(student_output)
                teacher_output = transform(teacher_output)

            comp = self.kd_comparison(student_output, teacher_output)
            comp = recursive_apply(comp, lambda x: x.mean())
            if isinstance(comp, Sequence):
                comp = torch.stack(comp).sum()
            elif isinstance(comp, torch.Tensor):
                if comp.numel() > 1:
                    comp = torch.stack(comp.tolist()).sum()
                else:
                    comp = comp.item()
            # else float
            self.kd_last_comparison[0] = comp

        return org_output

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        return self.student_layer.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.student_layer.load_state_dict(state_dict, strict=strict)

    def _named_members(self, get_members_fn, prefix="", recurse=True, **kwargs):
        for name, module in self.student_layer._named_members(
            get_members_fn, prefix=prefix, recurse=recurse, **kwargs
        ):
            yield name, module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        return self.student_layer.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )
