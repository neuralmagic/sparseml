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
from typing import Callable, List, Optional

import torch
from torch.nn import Module

from sparseml.core import State
from sparseml.core.model.pytorch import ModifiableModelPyTorch
from sparseml.modifiers.smoothquant.base import SmoothQuantModifier, SmoothQuantScale
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward


_LOGGER = logging.getLogger(__name__)

MINIMUM_SMOOTHING_SCALE = 1e-5

__all__ = ["SmoothQuantModifierPyTorch"]


class SmoothQuantModifierPyTorch(SmoothQuantModifier):
    """
    PyTorch implementation of the SmoothQuant algorithm

    :param calibration_function: optional function to use for the forward pass, or None
    to use the default tensor_module_forward
    """

    calibration_function: Optional[Callable] = None
    hooks_: List = None
    device_: Optional[str] = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run SmoothQuant on the given state

        :param state: state to run SmoothQuant on
        :return: True on a successful run, False otherwise
        """
        super(SmoothQuantModifierPyTorch, self).on_initialize(state, **kwargs)

        calibration_dataloader = state.data.calib
        self.device_ = torch.device(state.hardware.device)
        self.hooks_ = []

        self._setup_scale_hooks()
        self._calibrate(state.model, calibration_dataloader)
        self._apply_smoothing()

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the CUDA cache

        :param state: unused
        :return: True
        """
        super(SmoothQuantModifierPyTorch, self).on_finalize(state, **kwargs)
        torch.cuda.empty_cache()

        return True

    def _setup_scale_hooks(self):
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def create_hook_fn(layer_name):
            def hook_fn(module, inp, out):
                # update the per-channel min/max output values seen during calibration
                if isinstance(out, tuple):
                    out = out[0]

                hidden_dim = out.shape[-1]
                out = out.view(-1, hidden_dim)
                latest_mins = torch.min(out, dim=0)[0]
                latest_maxes = torch.max(out, dim=0)[0]

                if layer_name in self.scales_:
                    self.scales_[layer_name].min_channel_vals = torch.minimum(
                        self.scales_[layer_name].min_channel_vals, latest_mins
                    )
                    self.scales_[layer_name].max_channel_vals = torch.maximum(
                        self.scales_[layer_name].max_channel_vals, latest_maxes
                    )
                else:
                    self.scales_[layer_name] = SmoothQuantScale(
                        min_channel_vals=latest_mins, max_channel_vals=latest_maxes
                    )

            return hook_fn

        for mapping in self.resolved_mappings_:
            name = mapping.smooth_name
            layer = mapping.smooth_layer
            self.hooks_.append(layer.register_forward_hook(create_hook_fn(name)))

    @torch.no_grad()
    def _calibrate(self, model: ModifiableModelPyTorch, calibration_dataloader: List):
        """
        Catch the output dynamic ranges of each layer that will be smoothed by running
        forward passes with calibration_dataloader
        """
        class_name = self.__class__.__name__.replace("PyTorch", "")
        _LOGGER.info(f"Running {class_name} scale calibration...")
        if not calibration_dataloader:
            raise ValueError(
                "Calibration data loader not set, must populate the calib_data field of"
                " SparseSession to run the SmoothQuant modifier"
            )

        run_calibration_forward(
            model.model,
            calibration_dataloader,
            self.num_calibration_steps,
            self.calibration_function,
            self.device_,
        )

        # remove the hooks now that we are done calibrating
        for hook in self.hooks_:
            hook.remove()
        del self.hooks_

    @torch.no_grad()
    def _apply_smoothing(self):
        """
        After calibration, apply smoothing to the activations and push the transform
        into the following weights by applying the inverse to each balance weight.

        Y = (Xdiag(scales)^(-1) * diag(scales)W) where W is the to_balance weights and
        X is the to_smooth weights

        This modifies the weights of the model in-place.
        """
        _LOGGER.info("Smoothing activation scales...")
        for mapping in self.resolved_mappings_:
            activation_scales = (  # get dynamic range for each activation channel
                self.scales_[mapping.smooth_name].max_channel_vals
                - self.scales_[mapping.smooth_name].min_channel_vals
            )
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers

            scales = self._calculate_smoothing_scales(balance_layers, activation_scales)
            scales = torch.maximum(
                scales, torch.Tensor([MINIMUM_SMOOTHING_SCALE]).to(scales.device)
            )

            # invert the smoothing in the following layers
            for layer in balance_layers:
                layer.weight.mul_(scales.view(1, -1))

            # apply the smoothing
            if smooth_layer.weight.ndim == 1:
                smooth_layer.weight.div_(scales)
            else:
                smooth_layer.weight.div_(scales.view(-1, 1))
            if hasattr(smooth_layer, "bias") and smooth_layer.bias is not None:
                smooth_layer.bias.div_(scales)

    def _calculate_smoothing_scales(
        self, balance_layers: List[Module], activation_scales: torch.Tensor
    ) -> List[float]:
        """
        Calculate how much smoothing to apply to each channel based on the dynamic
        range of the activation and the following weights

        :param balance_layers: layers to offset activation smoothing to
        :param activation_scales: channel-wise dynamic range of activations to smooth
        :return: channel-wise scales to use for smoothing activations
        """
        # get the channel-wise dynamic range for each layer to be balanced
        weight_scales = []
        for layer in balance_layers:
            scale = layer.weight.abs().max(dim=0, keepdim=True)[0]
            weight_scales.append(scale)
        weight_scales = 2.0 * torch.cat(weight_scales, dim=0).max(dim=0)[0]

        # calculate the amount of smoothing to apply
        # s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        # where j is the input channel, alpha is smoothing strength
        scales = activation_scales.pow(self.smoothing_strength) / weight_scales.pow(
            1 - self.smoothing_strength
        )
        scales = torch.where(weight_scales > 0.0, scales, activation_scales)
        return scales
