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
Implementations related to activations for neural networks in PyTorch
"""

from typing import Union

import torch.nn.functional as TF
from torch import Tensor, clamp
from torch.nn import LeakyReLU, Module, PReLU
from torch.nn import ReLU as TReLU
from torch.nn import ReLU6 as TReLU6


try:
    from torch.nn import SiLU
except ImportError:
    SiLU = None


__all__ = [
    "ReLU",
    "ReLU6",
    "Swish",
    "Hardswish",
    "swish",
    "hard_swish",
    "create_activation",
    "replace_activation",
    "replace_activations",
    "is_activation",
]


class ReLU(TReLU):
    """
    ReLU wrapper to enforce that number of channels for the layer is passed in.
    Useful for activation sparsity work.

    :param num_channels: number of channels for the layer
    :param inplace: True to run the operation in place in memory, False otherwise
    """

    def __init__(self, num_channels: int = -1, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.num_channels = num_channels


class ReLU6(TReLU6):
    """
    ReLU6 wrapper to enforce that number of channels for the layer is passed in.
    Useful for activation sparsity work.

    :param num_channels: number of channels for the layer
    :param inplace: True to run the operation in place in memory, False otherwise
    """

    def __init__(self, num_channels: int = -1, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.num_channels = num_channels


def swish(x_tens: Tensor):
    """
    Swish layer functional implementation: x * sigmoid(x).
    More information can be found in the paper
    `here <https://arxiv.org/abs/1710.05941>`__.

    :param x_tens: the input tensor to perform the swish op on
    :return: the output of x_tens * sigmoid(x_tens)
    """
    return x_tens * TF.sigmoid(x_tens)


class Swish(Module):
    """
    Swish layer OOP implementation: x * sigmoid(x).
    More information can be found in the paper
    `here <https://arxiv.org/abs/1710.05941>`__.

    :param num_channels: number of channels for the layer
    """

    def __init__(self, num_channels: int = -1):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, inp: Tensor):
        return swish(inp)


def hard_swish(x_tens: Tensor, inplace: bool = False):
    """
    | Hardswish layer implementation:
    |    0 for x <= -3
    |    x for x >= 3
    |    x * (x + 3) / 6 otherwise

    More information can be found in the paper
    `here <https://arxiv.org/abs/1905.02244>`__.

    :param x_tens: the input tensor to perform the swish op on
    :param inplace: True to run the operation in place in memory, False otherwise
    :return: 0 for x <= -3, x for x >= 3, x * (x + 3) / 6 otherwise
    """
    if inplace:
        x_tens.mul_(clamp(x_tens + 3, 0, 6))
        x_tens.div_(6)
    else:
        relu_6 = x_tens + 3
        relu_6 = relu_6.clamp(0, 6)
        x_tens = x_tens * relu_6
        x_tens = x_tens / 6
    return x_tens


class Hardswish(Module):
    """
    | Hardswish layer implementation:
    |    0 for x <= -3
    |    x for x >= 3
    |    x * (x + 3) / 6 otherwise

    More information can be found in the paper
    `here <https://arxiv.org/abs/1905.02244>`__.

    :param num_channels: number of channels for the layer
    :param inplace: True to run the operation in place in memory, False otherwise
    """

    def __init__(self, num_channels: int = -1, inplace: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.inplace = inplace

    def forward(self, inp: Tensor):
        return hard_swish(inp, self.inplace)


def replace_activation(
    module: Module,
    name: str,
    act_type: str,
    inplace: bool = False,
    num_channels: Union[int, None] = None,
    **kwargs,
) -> Module:
    """
    General function to replace the activation for a specific layer in a Module
    with a new one.

    :param module: the module to replace the activation function in
    :param name: the name of the layer to replace the activation for
    :param act_type: the type of activation to replace with; options:
        [relu, relu6, prelu, lrelu, swish, silu]
    :param inplace: True to create the activation as an inplace, False otherwise
    :param num_channels: The number of channels to create the activation for
    :param kwargs: Additional kwargs to pass to the activation constructor
    :return: the created activation layer
    """
    layer = module
    layers = name.split(".")

    for lay in layers[:-1]:
        layer = layer.__getattr__(lay)

    cur = layer.__getattr__(layers[-1])

    if num_channels is None and hasattr(cur, "num_channels"):
        num_channels = cur.num_channels
    elif num_channels is None and hasattr(cur, "num_parameters"):
        num_channels = cur.num_parameters

    act = create_activation(
        act_type, inplace=inplace, num_channels=num_channels, **kwargs
    )
    layer.__setattr__(layers[-1], act)

    return act


def replace_activations(
    module: Module,
    act_type: str,
    inplace: bool = False,
    num_channels: Union[int, None] = None,
    **kwargs,
) -> Module:
    """
    General function to replace all activation functions in a Module
    with a new one.

    :param module: the module to replace the activation function in
    :param act_type: the type of activation to replace with; options:
        [relu, relu6, prelu, lrelu, swish, silu]
    :param inplace: True to create the activation as an inplace, False otherwise
    :param num_channels: The number of channels to create the activation for
    :param kwargs: Additional kwargs to pass to the activation constructor
    :return: the updated module
    """
    if is_activation(module):
        return create_activation(
            act_type, inplace=inplace, num_channels=num_channels, **kwargs
        )

    for child_name, child_module in module.named_children():
        setattr(
            module,
            child_name,
            replace_activations(
                child_module, act_type, inplace, num_channels, **kwargs
            ),
        )

    return module


def create_activation(
    act_type: str, inplace: bool, num_channels: int, **kwargs
) -> Module:
    """
    Create an activation function using the given parameters.

    :param act_type: the type of activation to replace with; options:
        [relu, relu6, prelu, lrelu, swish, hardswish, silu]
    :param inplace: True to create the activation as an inplace, False otherwise
    :param num_channels: The number of channels to create the activation for
    :param kwargs: Additional kwargs to pass to the activation constructor
    :return: the created activation layer
    """
    act_type = act_type.lower()

    if act_type == "relu":
        return ReLU(num_channels=num_channels, inplace=inplace)

    if act_type == "relu6":
        return ReLU6(num_channels=num_channels, inplace=inplace)

    if act_type == "prelu":
        return PReLU(num_parameters=num_channels, **kwargs)

    if act_type == "lrelu":
        return LeakyReLU(inplace=inplace, **kwargs)

    if act_type == "swish":
        return Swish(num_channels=num_channels)

    if act_type == "hardswish":
        return Hardswish(num_channels=num_channels, inplace=inplace)

    if act_type == "silu":
        return SiLU(**kwargs)

    raise ValueError("unknown act_type given of {}".format(act_type))


def is_activation(module: Module) -> bool:
    """
    :param module: the module to check whether it is a common activation function or not
    :return: True if the module is an instance of a common activation function,
        False otherwise
    """
    return (
        isinstance(module, TReLU)
        or isinstance(module, TReLU6)
        or isinstance(module, ReLU)
        or isinstance(module, ReLU6)
        or isinstance(module, PReLU)
        or isinstance(module, LeakyReLU)
        or isinstance(module, Swish)
        or isinstance(module, Hardswish)
        or (SiLU is not None and isinstance(module, SiLU))
    )
