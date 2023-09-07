from typing import Dict, Tuple

import torch
import torch.nn as nn

from math import ceil

from sparseml.pytorch.optim.manager import ScheduledModifierManager


class ModelPreProcessor:
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


class QuantizationModelPreprocessor(ModelPreProcessor):
    def __init__(
            self,
            recipe: str,
            data_loader,
            observer_batches,
            model_eval,
    ):
        self.recipe = recipe
        if self.recipe is None:
            raise ValueError("Recipe must not be None")
        self.data_loader = data_loader
        self.observer_batches = observer_batches
        self.model_eval = model_eval

    def __call__(self, model, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        manager = ScheduledModifierManager.from_yaml(self.recipe)
        model.train()
        manager.apply_structure(model, epoch=0.1)
        model.eval()
        model = self.initialize_scales_from_batches(model, dev)
        return model, {"manager": manager}

    def initialize_scales_from_batches(self, model, dev):
        print("Collecting data statistics for quantization scales...")
        model.train()
        with torch.no_grad():
            for _ in range(int(ceil(self.observer_batches / len(self.dataloader)))):
                self.model_eval(model, self.dataloader, dev)
        model.apply(torch.quantization.disable_observer)
        model.eval()
        return model
