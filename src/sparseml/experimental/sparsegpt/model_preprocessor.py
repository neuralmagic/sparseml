from collections.abc import Mapping
from typing import Dict, Tuple

import torch
import torch.nn as nn

from smoothquant.smooth import smooth_lm
from sparseml.pytorch.optim.manager import ScheduledModifierManager


class ModelPreProcessor:
    def __init__(self, model):
        self.model = model

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}


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


class QuantizationModelPreprocessor(ModelPreProcessor):
    def __init__(self, recipe: str, data_loader, observer_batches):
        self.recipe = recipe
        if self.recipe is None:
            raise ValueError("Recipe must not be None")
        self.data_loader = data_loader
        self.observer_batches = observer_batches

    def __call__(self, model, dev: str = "cuda:0") -> Tuple[nn.Module, Dict]:
        manager = ScheduledModifierManager.from_yaml(self.recipe)
        model.train()
        manager.apply_structure(model, epoch=0.1)
        model.eval()
        model = self.initialize_scales_from_batches(model, dev)
        return model, {"manager": manager}

    def initialize_scales_from_batches_whole(self, model, dev):
        print("Collecting data statistics for quantization scales...")
        model.train()
        model.to(dev)
        with torch.no_grad():
            batches = 0
            while batches < self.observer_batches:
                for batch in self.data_loader:
                    if batches == self.observer_batches:
                        break
                    print(f"Batch {batches + 1}/{self.observer_batches}")
                    if isinstance(batch, tuple):
                        inp, _ = batch  # Ignore target
                        inp = inp.to(dev)
                    elif isinstance(batch, Mapping):
                        if "labels" in batch:
                            batch.pop("labels")
                        inp = {k: v.to(dev) for k, v in batch.items()}
                    else:
                        raise ValueError(
                            f"Dont know how to process given batch type: {type(batch)}"
                        )
                    model(inp)
                    batches += 1
        model.apply(torch.quantization.disable_observer)
        return model
