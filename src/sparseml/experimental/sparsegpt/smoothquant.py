import torch
from typing import Dict, Tuple

BALANCING_MODULES = [
    torch.ao.quantization.QuantWrapper,
]

MERGING_MODULES = [
    torch.ao.quantization.QuantWrapper,
    torch.nn.LayerNorm,
    torch.nn.Linear,
]

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
            if isinstance(module, module_type) and hasattr(module.module, "weight"):
                return True
        return False

    def is_merge_module(self, module, merge_module_type):
        for module_type in MERGING_MODULES:
            if isinstance(module, module_type) or module.__class__.__name__.lower() == merge_module_type:
                return True
        return False

    def get_smoothquant_subgraph(self, layer, subgraph_keys):
        module_to_merge_scale = None
        modules_to_balance = []
        for name, child in layer.named_modules():
            for key in subgraph_keys["module_to_balance"]:
                if key in name and self.is_balance_module(child):
                    modules_to_balance.append(child)
            if (
                    subgraph_keys["module_to_merge_scale"] is not None
                    and subgraph_keys["module_to_merge_scale"][0] in name
                    and self.is_merge_module(child, subgraph_keys["module_to_merge_scale"][1])
            ):
                module_to_merge_scale = child
        return modules_to_balance, module_to_merge_scale

    @torch.no_grad()
    def apply_smoothing(self, layer):
        for subgraph_keys in self.subgraph_keys:
            modules_to_balance, module_to_merge_scale = self.get_smoothquant_subgraph(layer, subgraph_keys)
            activation_scale = modules_to_balance[0].quant.activation_post_process.scale
            weight_scales = []
            for b in modules_to_balance:
                weight_scales.append(torch.max(torch.abs(b.module.weight), keepdim=True, dim=0)[0])
            weight_scales = torch.cat(weight_scales, dim=0)
            weight_scales = torch.max(weight_scales, dim=0)[0] / 128.

            scales = activation_scale.pow(self.alpha) / weight_scales.pow(1.0 - self.alpha)

            for b in modules_to_balance:
                b.module.weight.mul_(scales.view(1, -1))

            if module_to_merge_scale is not None:
                if hasattr(module_to_merge_scale, "weight") and module_to_merge_scale.weight is not None:
                    if module_to_merge_scale.weight.ndim == 1:
                        module_to_merge_scale.weight.div_(scales)
                    else:
                        module_to_merge_scale.weight.div_(scales.view(-1, 1))
                elif (
                        hasattr(module_to_merge_scale, "module")
                        and hasattr(module_to_merge_scale.module, "weight")
                        and module_to_merge_scale.module.weight is not None
                ):
                    if module_to_merge_scale.module.weight.ndim == 1:
                        module_to_merge_scale.module.weight.div_(scales)
                    else:
                        module_to_merge_scale.module.weight.div_(scales.view(-1, 1))

                if hasattr(module_to_merge_scale, "bias") and module_to_merge_scale.bias is not None:
                    module_to_merge_scale.bias.div_(scales)
                elif (
                        hasattr(module_to_merge_scale, "module")
                        and hasattr(module_to_merge_scale.module, "bias")
                        and module_to_merge_scale.module.bias is not None
                ):
                    module_to_merge_scale.module.bias.div_(scales)

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[torch.nn.Module, Dict]:
        for layer in self.layers:
            self.apply_smoothing(layer)