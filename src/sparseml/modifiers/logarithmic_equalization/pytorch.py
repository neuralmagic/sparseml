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

import logging
from typing import List

import torch
from torch.nn import Module

from sparseml.modifiers.smoothquant.pytorch import SmoothQuantModifierPyTorch


_LOGGER = logging.getLogger(__name__)

__all__ = ["LogarithmicEqualizationModifierPyTorch"]


class LogarithmicEqualizationModifierPyTorch(SmoothQuantModifierPyTorch):
    """
    PyTorch implementation of the Logarithmic Activation Equalization algorithm

    :param calibration_function: optional function to use for the forward pass, or None
    to use the default tensor_module_forward
    """

    def _calculate_smoothing_scales(
        self, balance_layers: List[Module], activation_scales: torch.Tensor
    ) -> List[float]:
        """
        Calculate how much smoothing to apply to each channel based on the dynamic
        range of the activations and the following weights.

        :param balance_layers: layers to offset activation smoothing to
        :param activation_scales: channel-wise dynamic range of activations to smooth
        :return: channel-wise scales to use for smoothing activations
        """
        # calculate the amount of smoothing to apply
        # s_j = max(|X_j|) / log2( 2 + max(|X_j|) )
        # where j is the input channel
        scales = activation_scales / torch.log2(2 + activation_scales)
        return scales
