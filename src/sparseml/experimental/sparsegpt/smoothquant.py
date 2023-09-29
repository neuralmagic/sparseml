import torch
from sparseml.experimental.sparsegpt.model_preprocessor import ModelPreprocessor
import difflib

class SmoothQuantModelPreprocessor(ModelPreprocessor):
    def __init__(
            self,
            model,
            data_loader,
            model_forward,
            layer_mappings,
            alpha: float = 0.5,
            logarithmic_equalization: bool = False,
    ):
        super().__init__(model)
        self.data_loader = data_loader
        self.model_forward = model_forward
        self.layer_mappings = layer_mappings
        self.alpha = alpha
        self.logarithmic_equalization = logarithmic_equalization
        self.resolved_mappings = None
        self.scales = None
        self.resolve_layer_mappings()

    def resolve_layer_mappings(self):
        self.resolved_mappings = []
        for mapping in self.layer_mappings:
            for module_suffix in mapping["module_to_merge"]:
                if module_suffix is not None:
                    modules_to_merge = self._get_modules_by_suffix(module_suffix)
                    for name, module in modules_to_merge:
                        modules_to_balance = []
                        for balance_suffix in mapping["module_to_balance"]:
                            modules_to_balance.append(self.get_module_by_suffix(balance_suffix, name))
                        mapping = {
                            "module_to_merge": (name, module),
                            "modules_to_balance": modules_to_balance,
                        }
                        self.resolved_mappings.append(mapping)


    def get_modules_by_suffix(self, module_suffix):
        matches = []
        for name, module in self.model.named_modules():
            if name.endswith(module_suffix):
                matches.append([(name, module)])
        return matches

    def get_module_by_suffix(self, module_suffix, name_to_match):
        named_modules = self._get_modules_by_suffix(module_suffix)
        highest_len_match = 0
        match = None
        for name, module in named_modules:
            s = difflib.SequenceMatcher(None, name, name_to_match)
            match_start, match_end, size = s.find_longest_match(0, len(name), 0, len(name_to_match))
            if match_end - match_start > highest_len_match:
                match = (name, module)
                highest_len_match = match_end - match_start

        return match

    def compute_scales(self, dev):
        self.scales = {}

        def create_hook_fn(name):
            def hook_fn(m, inp, out):
                detached_input = inp.deatch()
                _min = min(detached_input.max().item(), self.scales[name][0])
                _max = max(detached_input.max().item(), self.scales[name][1])
                self.scales[name] = [_min, _max, _max - _min]
            return hook_fn

        hooks = []
        for mapping in self.resolved_mappings:
            for name, module in mapping["modules_to_balance"]:
                hooks.append(module.register_forward_hook(create_hook_fn(name)))

        self.model.eval()
        with torch.no_grad():
            self.model_forward(self.model, self.data_loader, dev)

        del hooks

    def cleanup(self):
        self.scales.clear()
        self.resolved_mappings.clear()
        torch.cuda.empty_cache()

    def apply_smoothing(self, dev):
        for mapping in self.resolved_mappings:
            activation_scale = None
            for name, module in mapping["modules_to_balance"]:
                if activation_scale is None:
                    activation_scale = torch.Tensor(self.scales[name][2].to(dev))
                else:
                    activation_scale = torch.maximum(max(self.scales[name][2].to(dev), activation_scale))

            if self.logarithmic_equalization:
                scales = activation_scale / torch.log2(2 + activation_scale)
            else:
                weight_scales = []
                for name, module in mapping["modules_to_balance"]:
                    if hasattr(module, "weight"):
                        weight_scales.append(module.weight.abs().max(dim=0, keepdim=True)[0].to(dev))
                weight_scales = torch.cat(weight_scales, dim=0).max(dim=0)[0]
                scales = activation_scale.pow(self.alpha) / weight_scales.pow(1. - self.alpha)

            module_to_merge = mapping["module_to_merge"][1]

            if module_to_merge is not None:
                module_to_merge.to(dev)
                if hasattr(module_to_merge, "weight") and module_to_merge.weight is not None:
                    if module_to_merge.weight.ndim == 1:
                        module_to_merge.weight.div_(scales)
                    else:
                        module_to_merge.weight.div_(scales.view(-1, 1))
                elif (
                        hasattr(module_to_merge, "module")
                        and hasattr(module_to_merge.module, "weight")
                        and module_to_merge.module.weight is not None
                ):
                    if module_to_merge.module.weight.ndim == 1:
                        module_to_merge.module.weight.div_(scales)
                    else:
                        module_to_merge.module.weight.div_(scales.view(-1, 1))

                if hasattr(module_to_merge, "bias") and module_to_merge.bias is not None:
                    module_to_merge.bias.div_(scales)
                elif (
                        hasattr(module_to_merge, "module")
                        and hasattr(module_to_merge.module, "bias")
                        and module_to_merge.module.bias is not None
                ):
                    module_to_merge.module.bias.div_(scales)
                module_to_merge.cpu()

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        self.compute_scales(dev)
        self.apply_smoothing(dev)
        self.cleanup()
        return self.model, {}