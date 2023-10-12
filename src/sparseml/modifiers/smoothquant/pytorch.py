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

from sparseml.core import State
from sparseml.core.model.pytorch import ModifiableModelPyTorch
from sparseml.modifiers.smoothquant.base import (
    SmoothQuantMapping,
    SmoothQuantModifier,
    SmoothQuantScale,
)
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward
from sparseml.utils.pytorch import get_matching_layer


_LOGGER = logging.getLogger(__name__)

__all__ = ["SmoothQuantModifierPyTorch"]


class SmoothQuantModifierPyTorch(SmoothQuantModifier):
    """
    PyTorch implementation of the SmoothQuant algorithm

    :param calibration_function: optional function to use for the forward pass, or None
    to use the default tensor_module_forward
    """

    calibration_function: Optional[Callable] = None
    hooks_: List = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run SmoothQuant on the given state

        :param state: state to run SmoothQuant on
        :return: True on a successful run, False otherwise
        """
        super(SmoothQuantModifierPyTorch, self).on_initialize(state, **kwargs)

        calibration_dataloader = state.data.calib
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

    @torch.no_grad()
    def _resolve_mappings(self, model: ModifiableModelPyTorch):
        """
        Transforms the list of activations to smooth and their corresponding weights
        into SmoothQuantMapping objects, resolving regular expressions.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        resolved_mappings = []
        for to_balance, to_smooth in self.mappings:
            to_smooth_layers = model.get_layers(to_smooth)
            for layer_name, smooth_layer in to_smooth_layers.items():
                if layer_name not in self.ignore:
                    balance_layers = []
                    for balance_suffix in to_balance:
                        # find the submodule that matches the activation layer
                        _, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, model.model
                        )
                        if balance_layer:
                            balance_layers.append(balance_layer)
                    # each mapping can contain multiple layers to balance, but only
                    # one layer to smooth
                    mapping = SmoothQuantMapping(
                        layer_name, smooth_layer, balance_layers
                    )
                    resolved_mappings.append(mapping)
        return resolved_mappings

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
                out = out.view(-1, hidden_dim).abs()
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
        _LOGGER.info("Running SmoothQuant scale calibration...")
        if not calibration_dataloader:
            raise ValueError(
                "Calibration data loader not set, must populate the calib_data field of"
                " SparseSession to run the SmoothQuant modifier"
            )

        model.model.eval()

        run_calibration_forward(
            model.model,
            calibration_dataloader,
            self.num_calibration_steps,
            self.calibration_function,
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

            # get the channel-wise dynamic range for each layer to be balanced
            weight_scales = []
            for layer in balance_layers:
                scale = layer.weight.abs().max(dim=0, keepdim=True)[0]
                weight_scales.append(scale)
            weight_scales = 2.0 * torch.cat(weight_scales, dim=0).max(dim=0)[0]

            # calculate the amount of smoothing to apply
            # s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
            # where j is the input channel, alpha is migration strength
            scales = activation_scales.pow(self.migration_strength) / weight_scales.pow(
                1 - self.migration_strength
            )

            # invert the smoothing in the following layers
            for layer in balance_layers:
                layer.weight.mul_(scales.view(1, -1))

            # apply the smoothing
            if smooth_layer.weight.ndim == 1:
                smooth_layer.weight.div_(scales)
            else:
                smooth_layer.weight.div_(scales.view(-1, 1))
            if hasattr(smooth_layer, "bias"):
                smooth_layer.bias.div_(scales)
