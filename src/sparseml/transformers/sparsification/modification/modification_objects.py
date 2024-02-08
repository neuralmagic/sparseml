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
Set of helper objects that are used to modify
the HuggingFace transformer models
"""
from typing import Any

import torch


__all__ = ["QuantizableIdentity", "QuantizableMatMul"]

# def swap_submodule(module: Any, current_submodule, new_submodule):
#     """
#     Swap a submodule in a module with a new submodule
#
#     :param module: the module to swap the submodule in
#     :param current_submodule: the submodule to swap out
#     :param new_submodule: the new submodule to swap in
#     :return: the module with the swapped submodule
#     """
#     pass
# #
# def replace_layer(module):
#     if isinstance(module, nn.Dropout):
#         return nn.Dropout(0)
#     elif isinstance(module, nn.Linear):
#         target_state_dict = deepcopy(module.state_dict())
#         bias = True if module.bias is not None else False
#         new_module = MixLinear(
#             module.in_features,
#             module.out_features,
#             bias,
#             target_state_dict['weight'],
#             0.9
#         )
#         new_module.load_state_dict(target_state_dict)
#         return new_module
#     else:
#         return module


def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


class QuantizableIdentity(torch.nn.Module):
    def forward(self, x):
        return x


class QuantizableMatMul(torch.nn.Module):
    """
    Wrapper around torch.matmul with distinct inputs/output class
    instances that could be quantized through SparseML recipe
    """

    def __init__(self, left_input_cls, right_input_cls):
        super().__init__()
        self.left_input = left_input_cls()
        self.right_input = right_input_cls()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(self.left_input(a), self.right_input(b))
