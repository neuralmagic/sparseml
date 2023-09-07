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

from typing import Dict, Tuple

import torch
from torch.nn import Module, Parameter

from sparseml.core import Event, State, EventType
from sparseml.modifiers.pruning.constant.base import ConstantPruningModifier


class ConstantPruningModifierPyTorch(ConstantPruningModifier):
    _layers_params: Dict[str, Tuple[Module, str, Parameter]] = None
    _forward_hooks = None
    _backward_hooks = None

    _save_masks: bool = False
    _use_hooks: bool = False
    _hooks_set: bool = False

    def on_initialize(self, state: State, event: Event, **kwargs) -> bool:
        if "save_masks" in kwargs:
            self._save_masks = kwargs["save_masks"]
        if "use_hooks" in kwargs:
            self._use_hooks = kwargs["use_hooks"]

        if not state.model or not state.start_event:
            return False

        self._layers_params = state.model.get_layers_params(self.targets)
        self._create_masks()
        self._check_create_hooks()

        return True

    def on_finalize(self, state: State, event: Event, **kwargs) -> bool:
        self._check_remove_masks()
        self._check_remove_hooks()

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self._populate_masks()

    def on_update(self, state: State, event: Event, **kwargs):
        if self._use_hooks:
            # hooks are used to update, so nothing to do here
            return

        if event.type_ == EventType.OPTIM_PRE_STEP:
            # zero out the gradients for the pruned params
            self._apply_mask_gradients()
        elif event.type_ == EventType.OPTIM_POST_STEP:
            # apply the masks to the pruned params
            self._apply_mask_params()

    def on_end(self, state: State, event: Event, **kwargs):
        self._check_remove_hooks()
        
    def _param_mask_name(self, param_name: str) -> str:
        return f"{param_name}_mask"

    def _create_masks(self):
        for name, (layer, param_name, param) in self._layers_params.items():
            # check if a mask is already applied to the layer
            try:
                layer.get_buffer(self._param_mask_name(param_name))
            except AttributeError:
                # add the mask buffer to the layer
                layer.register_buffer(
                    self._param_mask_name(param_name),
                    torch.ones_like(param.data, dtype=torch.bool),
                    persistent=self._save_masks,
                )

    def _populate_masks(self):
        for name, (layer, param_name, param) in self._layers_params.items():
            layer.get_buffer(self._param_mask_name(param_name)).fill_(
                param.data.abs() < self._epsilon
            )

    def _apply_mask_params(self):
        for name, (layer, param_name, param) in self._layers_params.items():
            mask = layer.get_buffer(self._param_mask_name(param_name))
            param.data = param.data * mask

    def _apply_mask_gradients(self):
        for name, (layer, param_name, param) in self._layers_params.items():
            if param.grad is not None:
                mask = layer.get_buffer(self._param_mask_name(param_name))
                param.grad = param.grad * mask

    def _check_remove_masks(self):
        if self._save_masks:
            return

        for name, (layer, param_name, param) in self._layers_params.items():
            try:
                layer.unregister_buffer(self._param_mask_name(param_name))
            except AttributeError:
                pass

    def _check_create_hooks(self):
        if not self._use_hooks or self._hooks_set:
            return

        def _register_hooks(layer, param_name, param):
            mask_name = self._param_mask_name(param_name)

            def _forward_hook_fn(module, input, output):
                mask = module.get_buffer(mask_name)
                param.data = param.data * mask

                return output

            def _backward_hook_fn(module, grad_input, grad_output):
                mask = module.get_buffer(mask_name)
                if grad_input[0] is not None:
                    grad_input[0] *= mask
                return grad_input

            forward_hook = layer.register_forward_hook(_forward_hook_fn)
            backward_hook = layer.register_backward_hook(_backward_hook_fn)

            return forward_hook, backward_hook

        self._forward_hooks = []
        self._backward_hooks = []

        for name, (layer, param_name, param) in self._layers_params.items():
            forward, backward = _register_hooks(layer, param_name, param)
            self._forward_hooks.append(forward)
            self._backward_hooks.append(backward)

        self._hooks_set = True

    def _check_remove_hooks(self):
        if self._hooks_set:
            return
        
        for forward, backward in zip(self._forward_hooks, self._backward_hooks):
            forward.remove()
            backward.remove()

        self._forward_hooks = None
        self._backward_hooks = None
        self._hooks_set = False
