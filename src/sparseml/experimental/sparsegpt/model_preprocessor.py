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

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.experimental.sparsegpt.dynamic_quantization import add_dynamic_quantization

from typing import Dict, Tuple

import torch
import torch.nn as nn

from math import ceil

from sparseml.experimental.sparsegpt.smoothquant import SmoothQuant

class ModelPreprocessor:
    def __init__(self, model):
        self.model = model

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}


def reset_observers(module):
    if hasattr(module, "reset_min_max_vals"):
        module.reset_min_max_vals()


def apply_recipe(model, recipe, dynamic_quantization_modules):
    manager = ScheduledModifierManager.from_yaml(recipe)
    model.train()
    manager.apply_structure(model, epoch=0.1)
    model.eval()
    if dynamic_quantization_modules is not None:
        add_dynamic_quantization(model, dynamic_quantization_modules)

    return manager


class QuantizationModelPreprocessor(ModelPreprocessor):
    def __init__(
            self,
            model,
            recipe: str,
            data_loader,
            observer_batches,
            model_forward,
            smoothquant=False,
            smoothquant_kwargs=None,
            dynamic_quantization_modules=None
    ):
        super().__init__(model)
        self.recipe = recipe
        if self.recipe is None:
            raise ValueError("Recipe must not be None")
        self.data_loader = data_loader
        self.observer_batches = observer_batches
        self.model_forward = model_forward
        self.smoothquant = smoothquant
        if smoothquant:
            self.smoothquant_instance = SmoothQuant(self.smoothquant_layers(), **smoothquant_kwargs)
        self.dynamic_quantization_modules = dynamic_quantization_modules

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        manager = apply_recipe(self.model, self.recipe, self.dynamic_quantization_modules)
        self.initialize_scales_from_batches(dev)
        if self.smoothquant:
            self.smoothquant_instance(dev)
            self.model.apply(reset_observers)
            self.initialize_scales_from_batches(dev)
        self.model.apply(torch.quantization.disable_observer)

        return self.model, {"manager": manager}


    def initialize_scales_from_batches(self, dev):
        print("Collecting data statistics for quantization scales...")
        self.model.eval()
        with torch.no_grad():
            for _ in range(int(ceil(self.observer_batches / len(self.data_loader)))):
                self.model_forward(self.model, self.data_loader, dev)

    def smoothquant_layers(self):
        pass

