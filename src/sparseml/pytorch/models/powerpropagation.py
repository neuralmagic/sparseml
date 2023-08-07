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

from typing import List, Tuple, Union

import torch
from torch import Tensor, abs, pow
from torch.nn import Conv2d, Linear, Module, Parameter
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


__all__ = [
    "convert_to_powerpropagation",
    "PowerPropagatedConv2d",
    "PowerPropagatedLinear",
]


class PowerPropagatedConv2d(Conv2d):
    def __init__(
        self,
        alpha,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = 1,
        padding: Union[str, Tuple[int, ...]] = 0,
        dilation: Tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:

        super(PowerPropagatedConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(
            input, self.weight * pow(abs(self.weight), self.alpha - 1), self.bias
        )

    def set_alpha(self, new_alpha):
        with torch.no_grad():
            self.weight *= pow(abs(self.weight), new_alpha / self.alpha - 1)
            self.alpha = new_alpha


class PowerPropagatedLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha=1,
        device=None,
        dtype=None,
    ) -> None:
        super(PowerPropagatedLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input, self.weight * pow(abs(self.weight), self.alpha - 1), self.bias
        )

    def set_alpha(self, new_alpha):
        with torch.no_grad():
            self.weight *= pow(abs(self.weight), new_alpha / self.alpha - 1)
            self.alpha = new_alpha


def convert_to_powerpropagation(module, device):
    module_output = module
    if type(module) is Conv2d:
        with torch.no_grad():
            module_output = PowerPropagatedConv2d(
                1,
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode,
                module.weight.device,
                module.weight.dtype,
            )
            module_output.weight = module.weight
            module_output.bias = module.bias
    elif type(module) is Linear:
        with torch.no_grad():
            module_output = PowerPropagatedLinear(
                1,
                module.in_features,
                module.out_features,
                module.bias is not None,
                module.weight.device,
                module.weight.dtype,
            )
            module_output.weight = module.weight
            module_output.bias = module.bias
    # If there are some layers that are sublcasses of Conv2d or Linear (maybe something
    # quantized, we can't handle that correctly and should throw an error.
    elif isinstance(module, Conv2d):
        raise RuntimeError("Can't convert a subclass of Conv2d to powerpropagation")
    elif isinstance(module, Linear):
        raise RuntimeError("Can't convert a subclass of Linear to powerpropagation")

    for name, child in module.named_children():
        module_output.add_module(name, convert_to_powerpropagation(child, device))
    return module_output
