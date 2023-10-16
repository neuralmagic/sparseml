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

import difflib
from typing import Dict, Tuple

import torch

from sparseml.experimental.sparsegpt.model_preprocessor import ModelPreprocessor


class SmoothQuantModelPreprocessor(ModelPreprocessor):
    def __init__(
        self,
        model,
        data_loader,
        model_forward,
        layer_mappings,
        ignore=None,
        alpha: float = 0.5,
    ):
        super().__init__(model)
        self.data_loader = data_loader
        self.model_forward = model_forward
        self.layer_mappings = layer_mappings
        self.ignore = ignore
        self.alpha = alpha
        self.resolved_mappings = None
        self.scales = None
        self.resolve_layer_mappings()

    def resolve_layer_mappings(self):
        self.resolved_mappings = []
        for mapping in self.layer_mappings:
            module_suffix = mapping["module_to_merge"]
            if module_suffix is not None:
                modules_to_merge = self.get_modules_by_suffix(module_suffix)
                for name, module in modules_to_merge:
                    if self.ignore is not None and name in self.ignore:
                        continue
                    modules_to_balance = []
                    for balance_suffix in mapping["module_to_balance"]:
                        modules_to_balance.append(
                            self.get_module_by_suffix(balance_suffix, name)
                        )
                    resolved_mapping = {
                        "module_to_merge": (name, module),
                        "module_to_balance": modules_to_balance,
                    }
                    self.resolved_mappings.append(resolved_mapping)

    def get_modules_by_suffix(self, module_suffix):
        matches = []
        for name, module in self.model.named_modules():
            if name.endswith(module_suffix):
                matches.append((name, module))
        return matches

    def get_module_by_suffix(self, module_suffix, name_to_match):
        named_modules = self.get_modules_by_suffix(module_suffix)
        highest_len_match = 0
        match = None
        for name, module in named_modules:
            s = difflib.SequenceMatcher(None, name, name_to_match)
            _, _, len_match = s.find_longest_match(0, len(name), 0, len(name_to_match))
            if len_match > highest_len_match:
                match = (name, module)
                highest_len_match = len_match

        return match

    def compute_scales(self, dev):
        self.scales = {}

        def create_hook_fn(name):
            def hook_fn(m, inp, out):
                if isinstance(out, tuple):
                    out = out[0]

                hidden_dim = out.shape[-1]
                detached_output = out.detach()
                detached_output = detached_output.view(-1, hidden_dim).abs().detach()
                coming_min = torch.min(detached_output, dim=0)[0]
                coming_max = torch.max(detached_output, dim=0)[0]
                if name in self.scales:
                    self.scales[name][0] = torch.minimum(
                        self.scales[name][0], coming_min
                    )
                    self.scales[name][1] = torch.maximum(
                        self.scales[name][1], coming_max
                    )
                    self.scales[name][2] = self.scales[name][1] - self.scales[name][0]
                else:
                    self.scales[name] = [
                        coming_min,
                        coming_max,
                        coming_max - coming_min,
                    ]

            return hook_fn

        hooks = []
        for mapping in self.resolved_mappings:
            name, module = mapping["module_to_merge"]
            hooks.append(module.register_forward_hook(create_hook_fn(name)))

        self.model.eval()
        with torch.no_grad():
            self.model_forward(self.model, self.data_loader, dev)

        del hooks

    def compute_balancing_scales(self, mapping, dev):
        name, module_to_merge = mapping["module_to_merge"]
        activation_scales = self.scales[name][2].to(dev)

        weight_scales = []
        for name, module in mapping["module_to_balance"]:
            if hasattr(module, "weight"):
                weight_scales.append(
                    module.weight.abs().max(dim=0, keepdim=True)[0].to(dev)
                )
        weight_scales = 2.0 * torch.cat(weight_scales, dim=0).max(dim=0)[0]
        return activation_scales.pow(self.alpha) / weight_scales.pow(
            1.0 - self.alpha
        )

    def apply_smoothing(self, dev):
        for mapping in self.resolved_mappings:
            scales = self.compute_balancing_scales(mapping, dev)
            module_to_merge = mapping["module_to_merge"][1]

            for _, module in mapping["module_to_balance"]:
                if hasattr(module, "weight"):
                    module.to(dev)
                    module.weight.mul_(scales.view(1, -1))
                    module.cpu()

            if module_to_merge is not None:
                module_to_merge.to(dev)
                if (
                    hasattr(module_to_merge, "weight")
                    and module_to_merge.weight is not None
                ):
                    if module_to_merge.weight.ndim == 1:
                        module_to_merge.weight.div_(scales)
                    else:
                        module_to_merge.weight.div_(scales.view(-1, 1))
                if (
                    hasattr(module_to_merge, "bias")
                    and module_to_merge.bias is not None
                ):
                    module_to_merge.bias.div_(scales)
                module_to_merge.cpu()

    def cleanup(self):
        self.scales.clear()
        self.resolved_mappings.clear()
        torch.cuda.empty_cache()

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[torch.nn.Module, Dict]:
        print("Collecting data statistics for smoothquant scales...")
        self.compute_scales(dev)
        self.apply_smoothing(dev)
        self.cleanup()
        return self.model, {}


class LogarithmicEqualizationPreprocessor(SmoothQuantModelPreprocessor):
    def compute_balancing_scales(self, mapping, dev):
        name, module_to_merge = mapping["module_to_merge"]
        activation_scales = self.scales[name][2].to(dev)
        return activation_scales / torch.log2(2 + activation_scales)
