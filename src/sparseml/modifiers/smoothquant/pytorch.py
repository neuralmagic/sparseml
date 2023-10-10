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
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Callable, Dict, List

import torch
from torch.nn import Module

from sparseml.core import Event, State
from sparseml.core.model.pytorch import ModifiableModelPyTorch
from sparseml.modifiers.smoothquant.base import SmoothQuantModifier
from sparseml.pytorch.utils import tensors_module_forward, tensors_to_device
from sparseml.utils.pytorch import get_matching_layer


_LOGGER = logging.getLogger(__name__)


@dataclass
class SmoothQuantScale:
    min_channel_vals: torch.Tensor
    max_channel_vals: torch.Tensor


@dataclass
class SmoothQuantMapping:
    merge_name: str
    merge_layer: Module
    balance_layers: List[Module]


class SmoothQuantModifierPyTorch(SmoothQuantModifier):
    calibration_dataloader_: Any = None
    calibration_function_: Any = None
    scales_: Dict = None
    hooks_: List = None
    resolved_mappings_: Dict = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.end and self.end != -1:
            raise ValueError(
                "SmoothQuantModifier can only be applied during one-shot. Expected end"
                " to be None or -1, got {}".format(self.end)
            )
        if self.start and self.start != -1:
            raise ValueError(
                "SmoothQuantModifier can only be applied during one-shot. Expected "
                "start to be None or -1, got {}".format(self.start)
            )

        self.calibration_dataloader_ = state.data.calib
        self.ignore = [] if not self.ignore else self.ignore
        self.scales_ = {}
        self.hooks_ = []

        self.resolved_mappings_ = self._resolve_mappings(state.model)
        self._setup_scale_hooks()
        self._calibrate(state.model)
        self._apply_smoothing()

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        self.scales_.clear()
        self.resolved_mappings_.clear()
        torch.cuda.empty_cache()

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    @torch.no_grad()
    def _resolve_mappings(self, model: ModifiableModelPyTorch):
        resolved_mappings = []
        for to_balance, to_merge in self.mappings:
            to_merge_layers = model.get_layers(to_merge)
            for layer_name, merge_layer in to_merge_layers.items():
                if layer_name not in self.ignore:
                    balance_layers = []
                    for balance_suffix in to_balance:
                        _, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, model.model
                        )
                        if balance_layer:
                            balance_layers.append(balance_layer)
                    mapping = SmoothQuantMapping(
                        layer_name, merge_layer, balance_layers
                    )
                    resolved_mappings.append(mapping)
        return resolved_mappings

    def _setup_scale_hooks(self):
        def create_hook_fn(layer_name):
            def hook_fn(module, inp, out):
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
            name = mapping.merge_name
            layer = mapping.merge_layer
            self.hooks_.append(layer.register_forward_hook(create_hook_fn(name)))

    @torch.no_grad()
    def _calibrate(self, model: ModifiableModelPyTorch):
        _LOGGER.info("Running SmoothQuant scale calibration...")
        if not self.calibration_dataloader_:
            raise ValueError(
                "Calibration data loader not set, must populate the calib_data field of"
                " SparseSession to run the SmoothQuant modifier"
            )

        model.model.eval()

        forward_fn: Callable = (
            self.calibration_function_
            if self.calibration_function_
            else tensors_module_forward
        )

        model_device = next(model.model.parameters()).device
        _dataloader = (
            self.calibration_dataloader_
            if self.num_calibration_steps is None
            else cycle(self.calibration_dataloader_)
        )

        for batch_idx, batch in enumerate(_dataloader):
            if self.num_calibration_steps and batch_idx >= self.num_calibration_steps:
                break
            batch = tensors_to_device(batch, model_device)
            with torch.no_grad():
                forward_fn(batch, module=model.model)

        del self.hooks_

    @torch.no_grad()
    def _apply_smoothing(self):
        _LOGGER.info("Smoothing activation scales...")
        for mapping in self.resolved_mappings_:
            activation_scales = (
                self.scales_[mapping.merge_name].max_channel_vals
                - self.scales_[mapping.merge_name].min_channel_vals
            )
            merge_layer = mapping.merge_layer
            balance_layers = mapping.balance_layers

            weight_scales = []
            for layer in balance_layers:
                scale = layer.weight.abs().max(dim=0, keepdim=True)[0]
                weight_scales.append(scale)
            weight_scales = 2.0 * torch.cat(weight_scales, dim=0).max(dim=0)[0]
            scales = activation_scales.pow(self.migration_strength) / weight_scales.pow(
                1 - self.migration_strength
            )

            for layer in balance_layers:
                layer.weight.mul_(scales.view(1, -1))

            if merge_layer.weight.ndim == 1:
                merge_layer.weight.div_(scales)
            else:
                merge_layer.weight.div_(scales.view(-1, 1))
            if hasattr(merge_layer, "bias"):
                merge_layer.bias.div_(scales)
