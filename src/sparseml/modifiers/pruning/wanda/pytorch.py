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

import torch

from sparseml.core.state import State
from sparseml.modifiers.pruning.wanda.base import WandaPruningModifier
from sparseml.modifiers.pruning.wanda.utils.helpers import (
    find_layers,
    prepare_calibration_input,
)
from sparseml.modifiers.pruning.wanda.utils.wrapped_gpt import WrappedGPT


_LOGGER = logging.getLogger(__name__)


class WandaPruningModifierPyTorch(WandaPruningModifier):
    """
    PyTorch implementation of WandaPruningModifier
    """

    def on_initialize(self, state: State, **kwargs) -> bool:
        modifiable_model = state.model
        pytorch_model = modifiable_model.model
        use_cache = pytorch_model.config.use_cache

        # set use_cache to False to avoid OOM
        pytorch_model.config.use_cache = False

        _LOGGER.info("Preparing calibration data")
        calibration_dataloader = state.data.calib
        device = state.hardware.device
        pytorch_model.to(device)
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                pytorch_model, calibration_dataloader, device
            )

        layers = pytorch_model.model.layers
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(len(calibration_dataloader)):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            for h in handles:
                h.remove()
            if self.mask_structure == "unstructured":
                prune_n = prune_m = 0
            else:
                prune_n, prune_m = tuple(map(int, self.mask_structure.split(":")))

            for name in subset:
                _LOGGER.info(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  # initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    indices = sort_res[1][:, : int(W_metric.shape[1] * self.sparsity)]
                    W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  # set weights to zero

            for j in range(len(calibration_dataloader)):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps

        pytorch_model.config.use_cache = use_cache
        torch.cuda.empty_cache()
        return True

    def on_finalize(self, state: State, **kwargs):
        return True
