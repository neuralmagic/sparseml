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

from sparseml.core import State
from sparseml.core.event import Event, EventType
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

    calibration_dataloader_: Any = None
    calibration_function_: Any = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.apply_globally:
            raise NotImplementedError("global pruning not implemented yet for PyTorch")

        return super().on_initialize(state, **kwargs)

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

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.BATCH_START:
            sparsity = self.scheduler_function_(event, state)
            if sparsity != self.current_sparsity_:
                self.current_sparsity_ = sparsity

                for (
                    layer_param_name,
                    parameterized_layer,
                ) in self.parameterized_layers_.items():
                    mask = self.mask_creator_function_(
                        PruningMaskCreatorArgs(
                            parameter=parameterized_layer.param,
                            sparsity=sparsity,
                            # TODO: Update scores
                            scores=parameterized_layer.param.data.abs(),
                        )
                    )
                    self.update_mask(layer_param_name, mask)
        else:
            self._update_masks(event)
