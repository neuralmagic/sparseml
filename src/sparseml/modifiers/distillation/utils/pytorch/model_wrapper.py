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

from typing import Any, Dict, Optional, Set

import torch
from torch.nn import Module


__all__ = ["KDModelWrapper"]


class KDModelWrapper(Module):
    KD_LAST_COMPARISON = "kd_last_comparison"

    def __init__(
        self,
        student_model: Module,
        teacher_model: Module,
        wrappers: Dict[str, Any],
        comparison,
        fsdp_active: bool,
    ):
        super(KDModelWrapper, self).__init__()

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.wrappers = wrappers
        self.kd_comparison = comparison
        self._fsdp_active = fsdp_active
        self.kd_enabled = False
        self.register_buffer(self.KD_LAST_COMPARISON, torch.zeros(1, device="cpu"))
        self._init_called = True  # make sure this is last property to be set

        def _clear_missing_keys(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()

        self.register_load_state_dict_post_hook(_clear_missing_keys)

    def forward(self, *args, **kwargs):
        self.teacher_model.eval()
        if not self.kd_enabled:
            return self.student_model(*args, **kwargs)

        org_output = self.student_model(*args, **kwargs)
        with torch.no_grad():
            self.teacher_model(*args, **kwargs)

        layerwise_comps = []
        nonpad_tokens = kwargs["attention_mask"] == 1
        device = nonpad_tokens.device
        for key, (student_wrapper, teacher_wrapper) in self.wrappers.items():
            student_out = student_wrapper.kd_last_transformed.to(device)[nonpad_tokens]
            teacher_out = teacher_wrapper.kd_last_transformed.to(device)[nonpad_tokens]
            comp = self.kd_comparison(student_out, teacher_out)
            layerwise_comps.append(comp)

        setattr(self, self.KD_LAST_COMPARISON, torch.stack(layerwise_comps).mean())

        return org_output

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        return self.student_model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.student_model.load_state_dict(state_dict, strict=strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.student_model._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        # we want the full names of modules in two cases
        # 1. trainer initialization, so teacher is moved to the correct device. This is
        # caught by the kd_enabled flag, which is set when the modifier is started
        # 2. running in DataParallel (non-FSDP) mode so the replicate function can pick
        # up the teacher.
        if not self.kd_enabled or not self._fsdp_active:
            return super().named_modules(
                memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
            )

        return self.student_model.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.student_model, name)
