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


from typing import Any

import torch

from sparseml.core import State
from sparseml.core.event import Event
from sparseml.modifiers.pruning.magnitude.pytorch import MagnitudePruningModifierPyTorch
from sparseml.modifiers.pruning.utils.pytorch.mask_factory import PruningMaskCreatorArgs
from sparseml.modifiers.pruning.wanda.base import WandaPruningModifier


class WandaPruningModifierPyTorch(
    WandaPruningModifier, MagnitudePruningModifierPyTorch
):
    """
    PyTorch implementation of the Wanda pruning modifier.
    Implements the Wanda algorithm from the paper:
        - https://arxiv.org/abs/2306.11695: "A Simple
        and Effective Pruning Approach for
        Large Language Models"
    """

    # start = -1
    calibration_dataloader_: Any = None
    calibration_function_: Any = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.apply_globally:
            raise NotImplementedError("global pruning not implemented yet for PyTorch")

        super().on_initialize(state, **kwargs)
        if self.end and self.end != -1:
            raise ValueError(
                "end_epoch is disabled for WandaPruningModifier and can only be set to"
                " -1 or None. Given {}".format(self.end)
            )

        if _start := self.calculate_start() == -1:  # one-shot
            return self.one_shot(state)

        raise ValueError(f"start must be -1 for WandaPruningModifier. Given {_start}")

    def one_shot(self, state: State):
        """
        One-shot pruning implementation for Wanda.
        """
        self.on_start(state, state.start_event)
        # input_activations = self._collect_input_activations(state)

        # set the sparsity to the final sparsity
        sparsity = self.final_sparsity

        for (
            layer_param_name,
            parameterized_layer,
        ) in self.parameterized_layers_.items():
            mask = self.mask_creator_function_(
                PruningMaskCreatorArgs(
                    parameter=parameterized_layer.param,
                    sparsity=sparsity,
                    scores=parameterized_layer.param.data.abs(),
                )
            )
            self.update_mask(layer_param_name, mask)
            self.apply_mask_gradient(layer_param_name)
            self.apply_mask_weight(layer_param_name)

    def on_start(self, state: State, event: Event, **kwargs):
        sparsity = self.scheduler_function_(event, state)
        self.current_sparsity_ = sparsity

        for layer_param_name, parameterized_layer in self.parameterized_layers_.items():
            mask = self.mask_creator_function_(
                PruningMaskCreatorArgs(
                    parameter=parameterized_layer.param,
                    sparsity=sparsity,
                    # TODO: Update scores
                    scores=parameterized_layer.param.data.abs(),
                )
            )
            self.update_mask(layer_param_name, mask)

        self.enable_masks()

    def _get_wanda_scores(self, parameterized_layer, input_activations):
        """
        Calculate the Wanda scores for a given layer.
        """

        weight = parameterized_layer.param.data
        input_activation = input_activations[parameterized_layer.name]
        l2_norm = torch.sqrt(input_activation.scaler_row.reshape(-1, 1))
        return wanda_score(weight=weight, activation_l2_norm=l2_norm)

    def _collect_input_activations(state):
        """
        returns a dict mapping of the following form:

        layer_param_name -> input_activation

        where the input_activation must be euclidean norm (w/o
        sqrt) of the input to the layer, and the layer_param_name is
        the name of the layer parameter in the model.
        The euclidean norm does not need to
        """
        raise NotImplementedError("TODO: implement this")


def wanda_score(weight, activation_l2_norm):
    """
    Calculate the Wanda score for a given weight
    and scaler row.
    """
    return weight.abs() * activation_l2_norm
