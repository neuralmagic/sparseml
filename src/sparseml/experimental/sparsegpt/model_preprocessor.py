from collections.abc import Mapping
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


class QuantizationModelPreprocessor(ModelPreprocessor):
    def __init__(self, model, recipe: str, data_loader, observer_batches):
        super().__init__(model)
        self.recipe = recipe
        if self.recipe is None:
            raise ValueError("Recipe must not be None")
        self.data_loader = data_loader
        self.observer_batches = observer_batches

    def __call__(self, dev: str = "cuda:0", **kwargs) -> Tuple[nn.Module, Dict]:
        manager = ScheduledModifierManager.from_yaml(self.recipe)
        self.model.train()
        manager.apply_structure(self.model, epoch=0.1)
        self.model.eval()
        self.model = self._initialize_scales_from_batches(dev)
        return self.model, {"manager": manager}

    def _initialize_scales_from_batches(self, dev):
        print("Collecting data statistics for quantization scales...")
        self.model.train()

        # Tuan: If the model does not fit into the device,
        # we need a different version of this func to forward
        # the batches through the model layer by layer
        # See: https://github.com/neuralmagic/neuralmagicml/blob/tuan-falcon/research/sparsegpt/falcon/FalconPress-main/modelutils.py
        self.model.to(dev)

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
                    self.model(inp)
                    del inp
                    batches += 1
        self.model.apply(torch.quantization.disable_observer)
        torch.cuda.empty_cache()
        return self.model
