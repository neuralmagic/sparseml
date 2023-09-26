import torch
from sparseml.experimental.sparsegpt.utils import replace_module


class DynamicQuantizationWrapper(torch.nn.Module):
    def __init__(self, module, bits=8):
        super().__init__()
        self.module = module
        self.bits = bits
        self.qmin, self.qmax = self._quantization_limits()

    def _quantization_limits(self):
        qmin = -2**(self.bits - 1)
        qmax = 2**(self.bits - 1) - 1
        return qmin, qmax

    def _get_quantization_parameters(self, x):
        x_min = torch.min(x, dim=-1, keepdim=True)[0]
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        scale = (x_max - x_min) / (self.qmax - self.qmin)
        zero_point = torch.clamp(self.qmin - torch.round(x_min / scale), min=self.qmin, max=self.qmax)
        return scale, zero_point

    def _quantize(self, x, scale, zero_point=0.):
        return torch.clip(torch.round((x / scale) + zero_point), min=self.qmin, max=self.qmax)

    def _dequantize(self, x, scale, zero_point):
        return (x - zero_point) * scale

    def _fake_quantize(self, x):
        scale, zero_point = self._get_quantization_parameters(x)
        return self._dequantize(self._quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x, *args, **kwargs):
        q = self._fake_quantize(x)
        return self.module(q, *args, **kwargs)


def add_dynamic_quantization(model, module_names):
    for name, child in model.named_modules():
        for target_name in module_names:
            if target_name == name.split(".")[-1]:
                replace_module(model, child, DynamicQuantizationWrapper(child))
                break