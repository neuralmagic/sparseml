from collections.abc import Mapping
from typing import Dict, Tuple

import torch
import torch.nn as nn

from math import ceil

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.experimental.sparsegpt.smoothquant import SmoothQuant


class ModelPreprocessor:
    def __init__(self, model):
        self.model = model

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}


def reset_observers(module):
    if hasattr(module, "reset_min_max_vals"):
        module.reset_min_max_vals()


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

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        manager = ScheduledModifierManager.from_yaml(self.recipe)
        self.model.train()
        manager.apply_structure(self.model, epoch=0.1)
        self.model.eval()
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

