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

import contextlib
import sys

import torch
import transformers
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10


def autocast_smart_context_manager(self):
    """
    A helper wrapper that creates an appropriate context manager for `autocast` while
    feeding it the desired arguments, depending on the situation.
    """
    enabled = enabled = hasattr(self, "scaler") and self.scaler.is_enabled()
    if self.use_cuda_amp or self.use_cpu_amp:
        if is_torch_greater_or_equal_than_1_10:
            ctx_manager = (
                torch.cpu.amp.autocast(dtype=self.amp_dtype)
                if self.use_cpu_amp
                else torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=enabled)
            )
        else:
            ctx_manager = torch.cuda.amp.autocast(enabled=enabled)
    else:
        ctx_manager = (
            contextlib.nullcontext()
            if sys.version_info >= (3, 7)
            else contextlib.suppress()
        )

    return ctx_manager


transformers.trainer.Trainer.autocast_smart_context_manager = (
    autocast_smart_context_manager
)
