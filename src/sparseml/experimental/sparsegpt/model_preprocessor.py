from collections.abc import Mapping
from typing import Dict, Tuple

import torch
import torch.nn as nn

from math import ceil

from sparseml.pytorch.optim.manager import ScheduledModifierManager

BALANCING_MODULES = [
    nn.qat.Linear,
]

MERGING_MODULES = [
    nn.qat.Linear,
    nn.LayerNorm,
    nn.Linear,
]

class ModelPreprocessor:
    def __init__(self, model):
        self.model = model

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        return self.model, {}


class SmoothQuant:
    def __init__(
            self,
            layers,
            subgraph_keys,
            alpha: float = 0.5,
    ):
        self.layers = layers
        self.alpha = alpha
        self.subgraph_keys = subgraph_keys

    def is_balance_module(self, module):
        for module_type in BALANCING_MODULES:
            if isinstance(module, module_type):
                return True
        return False

    def is_merge_module(self, module):
        for module_type in MERGING_MODULES:
            if isinstance(module, module_type):
                return True
        return False

    def get_smoothquant_subgraph(self, layer, subgraph_keys):
        module_to_merge_scale = None
        for name, child in layer.named_modules():
            if subgraph_keys["module_to_balance"] in name and self.is_balance_module(child):
                module_to_balance = child
            elif (
                    subgraph_keys["module_to_merge_scale"] is not None
                    and subgraph_keys["module_to_merge_scale"] in name
                    and self.is_merge_module(child)
            ):
                module_to_merge_scale = child
        return module_to_balance, module_to_merge_scale, subgraph_keys["merge_scale"]

    def apply_smoothing(self, layer):
        for subgraph_keys in self.subgraph_keys:
            module_to_balance, module_to_merge_scale, merge_scale = self.get_smoothquant_subgraph(layer, subgraph_keys)
            activation_scale = module_to_balance.quant.activation_post_process.scale
            weight_scales = module_to_balance.module.weight_fake_quant.scale
            scales = activation_scale.pow(self.alpha) / weight_scales.pow(1.0 - self.alpha)
            scales = scales.clamp(min=1.e-5)

            module_to_balance.module.weight.mul_(scales.view(1, -1))
            module_to_balance.module_to_balance.quant.activation_post_process.scale.mul_(scales)
            if hasattr(module_to_balance.module, "bias"):
                module_to_balance.module.bias.mul_(scales.view(1, -1))

            if merge_scale and module_to_merge_scale is not None:
                if isinstance(module_to_merge_scale, torch.nn.LinearNorm):
                    module_to_merge_scale.weight.div_(scales)
                    module_to_merge_scale.bias.div_(scales)
                elif isinstance(module_to_merge_scale, torch.nn.qat.Linear):
                    module_to_merge_scale.module.weight.div_(scales.view(-1, 1))
                    module_to_merge_scale.module_to_balance.quant.activation_post_process.scale.div_(scales)
                    if hasattr(module_to_merge_scale.module, "bias"):
                        module_to_merge_scale.module.bias.div_(scales.view(-1, 1))
                elif isinstance(module_to_merge_scale, torch.nn.Linear):
                    module_to_merge_scale.weight.div_(scales.view(-1, 1))
                    if hasattr(module_to_merge_scale.module, "bias"):
                        module_to_merge_scale.bias.div_(scales.view(-1, 1))

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        for layer in self.layers:
            self.apply_smoothing(layer)


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
            self.smoothquant_instance()
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

