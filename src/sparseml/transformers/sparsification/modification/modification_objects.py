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

import torch


__all__ = ["QuantizableIdentity", "QuantizableMatMul", "swap_modules"]


def swap_modules(
    module: torch.nn.Module, submodule_name: str, submodule_to_replace: torch.nn.Module
):
    """
    Recursively unfold the submodules of the module according to the submodule_name
    to eventually replace the leaf submodule (accessed from the module through the
    submodule_name) with the submodule_to_replace.

    E.g
    ```
    swap_modules(module=Model,
                 module_name="layers.0.sublayer",
                 module_to_replace=ReplaceModule
                 )
    ```
    this will recursively call:
    1. SomeModule1 = getattr(Model, "layers")
    2. SomeModule2 = getattr(SomeModule1, "0")

    finally will swap the leaf submodule with the submodule_to_replace
    ```
    (submodule_name = "sublayer")
    setattr(SomeModule2 , submodule_name, ReplaceModule)
    ```
    this will essentially replace SomeModule2.sublayer with ReplaceModule

    :param module: the module to replace with the module_to_replace
    :param submodule_name: the name of the module to replace
    :param submodule_to_replace: the module to replace the module with
    """
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"module {module} is not a torch.nn.Module")
    if not isinstance(submodule_to_replace, torch.nn.Module):
        raise ValueError(
            f"submodule_to_replace {submodule_name} is not a torch.nn.Module"
        )

    attribute_name = submodule_name
    attribute_name = attribute_name.split(".", 1)
    if len(attribute_name) == 1:
        setattr(module, attribute_name[0], submodule_to_replace)
    else:
        swap_modules(
            getattr(module, attribute_name[0]), attribute_name[1], submodule_to_replace
        )


class QuantizableIdentity(torch.nn.Module):
    """
    Identity model that is introduced to be used
    together with QuantizableMatMul to allow for
    SparseML quantization scheme
    """

    def forward(self, x):
        return x


class QuantizableMatMul(torch.nn.Module):
    """
    Wrapper around torch.matmul with distinct inputs/output class
    instances that could be quantized through SparseML recipe

    :param left_input_cls: class instance that is used to quantize the left input
    :param right_input_cls: class instance that is used to quantize the right input
    :param output_cls: class instance that is used to quantize the output (optional)
    :return: the output of the matrix multiplication
    """

    def __init__(self, left_input_cls, right_input_cls, output_cls=None):
        super().__init__()
        self.left_input = left_input_cls()
        self.right_input = right_input_cls()
        self.output = output_cls() if output_cls is not None else None

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        out = torch.matmul(self.left_input(a), self.right_input(b))
        if self.output is not None:
            return self.output(out)
        return out
