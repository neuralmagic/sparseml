from collections.abc import Mapping
from typing import Dict, Tuple

import torch
import torch.nn as nn

from math import ceil

from sparseml.pytorch.optim.manager import ScheduledModifierManager


class ModelPreprocessor:
    def __init__(self, model):
        self.model = model

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}

try:
    from smoothquant.smooth import smooth_lm
    class SmoothQuantModelPreprocessor(ModelPreProcessor):
        def __init__(self, model, smooth_activation_file, alpha: float = 0.5):
            super().__init__(model)
            self.smooth_activation_file = smooth_activation_file
            self.alpha = alpha

        def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
            self.model.to(dev)
            act_scales = torch.load(self.smooth_activation_file)
            smooth_lm(self.model, act_scales, 0.5)
            return self.model
except:
    print("SmoothQuant not supported")


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
        manager = ScheduledModifierManager.from_yaml(self.recipe)
        self.model.train()
        manager.apply_structure(self.model, epoch=0.1)
        self.model.eval()
        model = self.initialize_scales_from_batches(dev)
        return model, {"manager": manager}

    def initialize_scales_from_batches(self, dev):
        print("Collecting data statistics for quantization scales...")
        self.model.eval()
        with torch.no_grad():
            for _ in range(int(ceil(self.observer_batches / len(self.data_loader)))):
                self.model_forward(self.model, self.data_loader, dev)
        self.model.apply(torch.quantization.disable_observer)
        return self.model
