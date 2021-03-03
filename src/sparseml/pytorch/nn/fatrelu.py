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
Implementations for the FATReLU (Forced Activation Threshold) activation function.
Used to increase the activation sparsity of neural networks.
"""

from typing import Dict, List, Union

import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module, Parameter, ReLU


__all__ = [
    "fat_relu",
    "fat_pw_relu",
    "fat_sig_relu",
    "fat_exp_relu",
    "FATReLU",
    "convert_relus_to_fat",
    "set_relu_to_fat",
]


def _apply_permuted_channels(apply_fn, tens: Tensor, **kwargs):
    if len(tens.shape) < 3:
        return apply_fn(tens, **kwargs)

    perm = [ind for ind in range(len(tens.shape))]
    # swap the channel and the last element so we can broadcast across the channels
    perm[1] = perm[-1]
    perm[-1] = 1

    return apply_fn(tens.permute(perm), **kwargs).permute(perm)


def fat_relu(tens: Tensor, threshold: Union[Tensor, float], inplace: bool) -> Tensor:
    """
    Apply a FATReLU function to a tensor (forced activation threshold):
    f(x, t) = 0 if x < t; x if x >= t

    :param tens: the tensor to apply the fat relu to
    :param threshold: the threshold to apply. if not a single value then
        the dimension to broadcast across must be last in the tensor
    :param inplace: False to create a new tensor,
        True to overwrite the current tensor's values
    :return: f(x, t) = 0 if x < t; x if x >= t
    """
    if isinstance(threshold, float):
        # not channelwise, can get by with using a threshold
        return TF.threshold(tens, threshold, 0.0, inplace)

    mask = (tens >= threshold).float()
    out = tens * mask if not inplace else tens.mul_(mask)

    return out


def fat_pw_relu(
    tens: Tensor, threshold: Tensor, compression: Tensor, inplace: bool
) -> Tensor:
    """
    Apply a piecewise separable FATReLU function to a tensor
    (forced activation threshold):
    f(x, t, c) = 0 if x <= (t - t/c); x if x >= t;
    c(x - (t - t/c)) if x > (t - t/c) and x < t

    :param tens: the tensor to apply the piecewise fat relu to
    :param threshold: the threshold at which all values will be zero or interpolated
        between threshold and 0
    :param compression: the compression or slope to interpolate between 0
        and the threshold with
    :param inplace: false to create a new tensor, true to overwrite the
        current tensor's values
    :return: f(x, t, c) = 0 if x <= (t - t/c); x if x >= t;
        c(x - (t - t/c)) if x > (t - t/c) and x < t
    """
    x_offset = threshold - threshold / compression

    # apply the fat relu up until our x_offset (where our compression region starts)
    out = fat_relu(tens, x_offset, inplace)

    # calculate the compression region values
    comp_mask = ((tens < threshold).float() * tens > x_offset).float()
    comp_tens = compression * (out - x_offset)

    # reassign the compression values in the output
    out = (
        (-1.0 * comp_mask + 1.0) * out + comp_tens * comp_mask
        if not inplace
        else out.mul_(-1.0 * comp_mask + 1.0).add_(comp_tens * comp_mask)
    )

    return out


def fat_sig_relu(tens: Tensor, threshold: Tensor, compression: Tensor) -> Tensor:
    """
    Create a sigmoid approximated FATReLU function to a tensor
    (forced activation threshold):
    f(x, t, c) = x / e^(c*(t-x))

    Note: there is no option for inplace with this function.

    :param tens: the tensor to apply the sigmoid fat relu to
    :param threshold: the threshold at which all values will be zero or approximated
        in the sigmoid region
    :param compression: the compression or slope to use in the sigmoid region
    :return: f(x, t, c) = x / e^(c*(t-x))
    """
    out = tens / (1.0 + torch.exp(compression * (threshold - tens)))
    out = TF.relu(
        out, inplace=True
    )  # make sure the negative region is always zero activation with a regular ReLU

    return out


def fat_exp_relu(tens: Tensor, threshold: Tensor, compression: Tensor) -> Tensor:
    """
    Create a piecewise separable exp approximated FATReLU function to a tensor
    (forced activation threshold):
    f(x, t, c) = 0 if x <= 0; = x if x >= t;
    = x * e^(c(x-t)) if x > 0 and x < t

    Note: there is no option for inplace with this function

    :param tens: the tensor to apply the exponential fat relu to
    :param threshold: the threshold at which all values will be zero or approximated
        in the exponential region
    :param compression: the compression or slope to use in the exponential region
    :return: f(x, t, c) = 0 if x <= 0; = x if x >= t;
        = x * e^(c(x-t)) if x > 0 and x < t
    """
    # remove the negative values
    out = TF.relu(tens)
    # calculate the compression region values
    comp_mask = ((out < threshold) * (out > 0.0)).float()

    comp_tens = out * torch.exp(compression * (out - threshold))

    # reassign the compression values in the output
    out = (-1.0 * comp_mask + 1.0) * out + comp_tens * comp_mask

    return out


class FATReLU(Module):
    """
    Applies a FAT ReLU (forced activation threshold) over the input.
    Instead of setting all negative values to 0 like with ReLU,
    this sets all values < threshold equal to 0

    :param threshold: the threshold that all values < threshold will be set to 0.
        if type float then f(x) = x if x >= threshold else 0.
        if type list then f(x[:, chan]) = x[:, chan]
        if x[:, chan] >= threshold[chan] else 0.
        if type list and empty, applies activation as the list option
        but dynamically initializes to the num chan
    :param inplace: perform the operation inplace or create a new tensor
    """

    def __init__(
        self, threshold: Union[float, List[float]] = 0.0, inplace: bool = False
    ):
        super(FATReLU, self).__init__()
        self._dynamic = False
        self._channel_wise = False
        self._num_channels = None

        if isinstance(threshold, List):
            self._channel_wise = True
            self._num_channels = len(threshold)

            if len(threshold) == 0:
                # can be dynamic only at init (before first data)
                # NB: _num_channles set dynamically - at first pass
                self._dynamic = True

        self.threshold = Parameter(torch.tensor(threshold))
        self.threshold.requires_grad = False
        self.inplace = inplace

    @property
    def dynamic(self) -> bool:
        """
        :return: True if the layer is in dynamic mode
            (gathering the number of channels), False otherwise
        """
        return self._dynamic

    @property
    def channel_wise(self) -> bool:
        """
        :return: True if the FATReLU is applied per channel, False otherwise
        """
        return self._channel_wise

    @property
    def num_channels(self):
        """
        :return: The number of channels the FATReLU is acting on
        """
        if self._dynamic:
            raise Exception(
                "number of channels not yet allocated. "
                "function should be called only after allocation"
            )

        return self._num_channels

    def set_threshold(self, threshold: Union[float, List[float]]):
        """
        :param threshold: the threshold value to set for the activation
        """
        if self._dynamic:
            raise RuntimeError(
                "cannot set threshold, threshold is setup activation dynamic "
                "(constructor given empty list)"
            )

        if self._channel_wise and isinstance(threshold, float):
            raise ValueError(
                "cannot set threshold to float value, "
                "constructor setup with list of channels len {}".format(
                    self._num_channels
                )
            )

        if self._channel_wise and self._num_channels != len(threshold):
            raise ValueError(
                "cannot set threshold to list of "
                "len({}), constructor setup with list of len({})".format(
                    len(threshold), self._num_channels
                )
            )

        current_tens = self.threshold.data  # type: Tensor
        new_tens = current_tens.new_tensor(threshold)
        current_tens.copy_(new_tens)

    def get_threshold(self) -> Union[float, List[float]]:
        """
        :return: the current threshold being applied for the activation
        """
        return (
            self.threshold.data.cpu().item()
            if not self._channel_wise
            else self.threshold.data.cpu().tolist()
        )

    def forward(self, inp: Tensor):
        if not self._channel_wise:
            threshold = self.threshold.data.item()

            return fat_relu(inp, threshold, self.inplace)

        if self._dynamic:
            thresh = [0.0] * inp.shape[1]
            self.threshold.data = torch.tensor(thresh)
            self._dynamic = False
            self._num_channels = len(thresh)

        assert (
            inp.shape[1] == self._num_channels
        )  # runtime test that #channels equals expected #channels

        return _apply_permuted_channels(
            fat_relu, inp, threshold=self.threshold, inplace=self.inplace
        )

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""

        return "threshold={}{}".format(self.threshold, inplace_str)

    def load_state_dict(self, state_dict, strict=True):
        if self._dynamic:
            raise Exception(
                "attempt to load state_dict, but fatrelu is not initialized yet."
                "need to pass data once to initialize channel since constructed "
                "with dynamic allocation of number of channels"
            )

        super().load_state_dict(state_dict, strict)


def convert_relus_to_fat(module: Module, **kwargs) -> Dict[str, FATReLU]:
    """
    Replace all of the ReLUs in a module with FATReLU instances.

    Note: only works if the ReLUs are layers in the module,
    will not work with torch.functional ones.

    :param module: the module to replace all ReLUs with FATReLU
    :param kwargs: the kwargs to pass to the FATReLU constructor
    :return: a dictionary containing a mapping from the names of the replaced layers
        to the replaced FATReLU
    """
    relu_keys = []

    for name, mod in module.named_modules():
        if isinstance(mod, ReLU):
            relu_keys.append(name)

    added = {}

    for key in relu_keys:
        added[key] = set_relu_to_fat(module, key, **kwargs)

    return added


def set_relu_to_fat(module: Module, layer_name: str, **kwargs) -> FATReLU:
    """
    Replace a given layer in a module to a FATReLU instance.

    :param module: the module to replace the given layer with a FATReLU implementation
    :param layer_name: the name of the layer to replace with a FATReLU
    :param kwargs: the kwargs to pass to the FATReLU constructor
    :return: the created FATReLU instance
    """
    layer = module
    layers = layer_name.split(".")

    for lay in layers[:-1]:
        layer = layer.__getattr__(lay)

    fat = layer.__getattr__(layers[-1])

    if not isinstance(fat, FATReLU):
        fat = FATReLU(**kwargs)

    layer.__setattr__(layers[-1], fat)

    return fat
