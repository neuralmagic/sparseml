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

from typing import Dict

from sparseml.core import Event, EventType, ModelParameterizedLayer, State
from sparseml.modifiers.pruning.helpers import (
    PruningCreateSettings,
    PruningSchedulerFactory,
    SchedulerCalculationType,
)
from sparseml.modifiers.pruning.magnitude.base import MagnitudePruningModifier
from sparseml.modifiers.pruning.utils.pytorch import (
    LayerParamMasking,
    MaskCreatorType,
    PruningMaskCreatorArgs,
    PruningMaskFactory,
)


class MagnitudePruningModifierPyTorch(MagnitudePruningModifier, LayerParamMasking):
    parameterized_layers_: Dict[str, ModelParameterizedLayer] = None
    _save_masks: bool = False
    _use_hooks: bool = False
    scheduler_function_: SchedulerCalculationType = None
    mask_creator_function_: MaskCreatorType = None
    current_sparsity_: float = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.apply_globally:
            raise NotImplementedError("global pruning not implemented yet for PyTorch")

        if "save_masks" in kwargs:
            self._save_masks = kwargs["save_masks"]
        if "use_hooks" in kwargs:
            self._use_hooks = kwargs["use_hooks"]

        if not state.model or not state.start_event:
            return False

        self.scheduler_function_ = PruningSchedulerFactory.create_scheduler(
            self.update_scheduler,
            PruningCreateSettings(
                self.start,
                self.end,
                self.update,
                self.init_sparsity,
                self.final_sparsity,
                self.scheduler_args,
            ),
        )
        self.mask_creator_function_ = PruningMaskFactory.create_mask_creator(
            self.mask_structure
        )

        self.parameterized_layers_ = state.model.get_layers_params(self.targets)

        for layer_param_name, parameterized_layer in self.parameterized_layers_.items():
            self.add_mask(
                layer_param_name,
                parameterized_layer,
                persistent=self._save_masks,
                add_hooks=self._use_hooks,
            )

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.leave_enabled:
            for layer_param_name, _ in self.parameterized_layers_.items():
                self.remove_mask(layer_param_name)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        sparsity = self.scheduler_function_(event, state)
        self.current_sparsity_ = sparsity

        for layer_param_name, parameterized_layer in self.parameterized_layers_.items():
            mask = self.mask_creator_function_(
                PruningMaskCreatorArgs(
                    parameter=parameterized_layer.param,
                    sparsity=sparsity,
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
                            scores=parameterized_layer.param.data.abs(),
                        )
                    )
                    self.update_mask(layer_param_name, mask)
        else:
            self._update_masks(event)

    def on_end(self, state: State, event: Event, **kwargs):
        if not self.leave_enabled:
            self.disable_masks()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.current_index >= self.end and self.leave_enabled:
            self._update_masks(event)

    def _update_masks(self, event: Event):
        if event.type_ == EventType.OPTIM_PRE_STEP and not self._use_hooks:
            for layer_param_name, _ in self.parameterized_layers_.items():
                self.apply_mask_gradient(layer_param_name)
        elif event.type_ == EventType.OPTIM_POST_STEP and not self._use_hooks:
            for layer_param_name, _ in self.parameterized_layers_.items():
                self.apply_mask_weight(layer_param_name)
