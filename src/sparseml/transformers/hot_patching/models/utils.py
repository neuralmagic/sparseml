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
Utils for hot patching specific models
"""
import functools
from typing import Callable

import torch


__all__ = ["QATOperation", "hf_forward_wrapper"]


class QATOperation(torch.nn.Module):
    def __init__(self, torch_op):
        super().__init__()
        # behaves like normal torch.matmul unless a SparseML QuantizationModifier
        # is initialized
        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 2,
            "input_qconfigs": ["asymmetric", "symmetric"],
        }
        self.forward = torch_op


def _torch_operation_patch(torch_op):
    torch_maybe_qat_op = QATOperation(torch_op)

    def wrapper(*args, **kwargs):
        return torch_maybe_qat_op(*args, **kwargs)

    return wrapper


def hf_forward_wrapper(forward_func: Callable, original_torch_func_path: str):
    """
    Drop in replacement for torch function with custom function, only during the scope
    of forward_func execution. e.g. replaces torch.matmul with
    QATOperation(torch.matmul) during execution of
    transformers.models.bert.modeling_bert.BertSelfAttention.forward
    """

    def forward_wrapped(self, *args, **kwargs):
        original_torch_func = _rgetattr(torch, original_torch_func_path)
        patched_torch_func = _torch_operation_patch(original_torch_func)
        _rsetattr(torch, original_torch_func_path, patched_torch_func)
        out = forward_func(self, *args, **kwargs)
        _rsetattr(torch, original_torch_func_path, original_torch_func)
        return out

    return forward_wrapped


def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
