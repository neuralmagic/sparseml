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

import torch

from sparseml.core import Event, EventType, ModelParameterizedLayer, State
from sparseml.modifiers.pruning.constant.base import ConstantPruningModifier
from sparseml.modifiers.pruning.utils.pytorch import LayerParamMasking, param_mask_name


class ConstantPruningModifierPyTorch(ConstantPruningModifier, LayerParamMasking):
    parameterized_layers_: Dict[str, ModelParameterizedLayer] = None
    _save_masks: bool = False
    _use_hooks: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        if "save_masks" in kwargs:
            self._save_masks = kwargs["save_masks"]
        if "use_hooks" in kwargs:
            self._use_hooks = kwargs["use_hooks"]

        if not state.model or not state.start_event:
            return False

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
        for layer_param_name, _ in self.parameterized_layers_.items():
            self.remove_mask(layer_param_name)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        for layer_param_name, parameterized_layer in self.parameterized_layers_.items():
            self.update_mask(
                layer_param_name, parameterized_layer.param.data.abs() > self._epsilon
            )

        self.enable_masks()

    @torch.no_grad()
    def on_update(self, state: State, event: Event, **kwargs):
        if self._use_hooks:
            # hooks are used to update, so nothing to do here
            return
        if event.type_ == EventType.OPTIM_POST_STEP:

            def apply_masks(module):
                mask_name = param_mask_name()
                if hasattr(module, mask_name):
                    mask = getattr(module, mask_name)
                    if mask.device != module.weight.device:
                        setattr(module, mask_name, mask.to(module.weight.device))
                    module.weight *= getattr(module, mask_name)

            state.model.model.apply(apply_masks)

    def on_end(self, state: State, event: Event, **kwargs):
        self.disable_masks()
