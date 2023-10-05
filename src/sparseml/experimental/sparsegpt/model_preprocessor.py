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

from math import ceil
from typing import Dict, Tuple

import torch
import torch.nn as nn

from sparseml.pytorch.optim.manager import ScheduledModifierManager


class ModelPreprocessor:
    def __init__(self, model):
        self.model = model

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}


class SmoothQuantModelPreprocessor(ModelPreprocessor):
    def __init__(self, model, smooth_activation_file, alpha: float = 0.5):
        super().__init__(model)
        self.smooth_activation_file = smooth_activation_file
        self.alpha = alpha

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        from smoothquant.smooth import smooth_lm

        self.model.to(dev)
        act_scales = torch.load(self.smooth_activation_file)
        smooth_lm(self.model, act_scales, 0.5)
        del act_scales
        torch.cuda.empty_cache()
        return self.model, {}


def apply_recipe(model, recipe):
    manager = ScheduledModifierManager.from_yaml(recipe)
    model.train()
    manager.apply_structure(model, epoch=0.1)
    model.eval()

    return manager


class QuantizationModelPreprocessor(ModelPreprocessor):
    def __init__(
        self,
        model,
        recipe: str,
        data_loader,
        observer_batches,
        model_forward,
    ):
        super().__init__(model)
        self.recipe = recipe
        if self.recipe is None:
            raise ValueError("Recipe must not be None")
        self.data_loader = data_loader
        self.observer_batches = observer_batches
        self.model_forward = model_forward

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        manager = apply_recipe(self.model, self.recipe)
        self.initialize_scales_from_batches(dev)
        self.model.apply(torch.quantization.disable_observer)
        return self.model, {"manager": manager}

    def initialize_scales_from_batches(self, dev):
        print("Collecting data statistics for quantization scales...")
        self.model.eval()
        with torch.no_grad():
            for _ in range(int(ceil(self.observer_batches / len(self.data_loader)))):
                self.model_forward(self.model, self.data_loader, dev)
