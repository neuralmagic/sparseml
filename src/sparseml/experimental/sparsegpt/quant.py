import torch
import torch.nn as nn


class WeightFakeQuantizer(nn.Module):
    def __init__(self, layer):
        super(WeightFakeQuantizer, self).__init__()
        if not isinstance(layer, nn.qat.Linear):
            raise ValueError("WeightFakeQuantizer expects Linear weight only")
        [
            setattr(self, attr, getattr(layer.weight_fake_quant, attr))
            for attr in ["scale", "zero_point", "dtype"]
        ]

    def quantize(self, w: torch.Tensor):
        q = torch.dequantize(
            torch.quantize_per_tensor(w, self.scale, self.zero_point, self.dtype)
        )
        return q


class MatMulLeftInput_QK(nn.Identity):
    ...


class MatMulRightInput_QK(nn.Identity):
    ...


class MatMulOutput_QK(nn.Identity):
    ...


class MatMulLeftInput_PV(nn.Identity):
    ...


class MatMulRightInput_PV(nn.Identity):
    ...


class MatMulOutput_PV(nn.Identity):
    ...


class QuantizableMatMul(nn.Module):
    """
    Wrapper around torch.matmul with distinct inputs/output class
    instances that could be quantized through SparseML recipe
    """

    def __init__(self, left_input_cls, right_input_cls, output_cls):
        super().__init__()
        self.left_input = left_input_cls()
        self.right_input = right_input_cls()
        self.output = output_cls()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.output(torch.matmul(self.left_input(a), self.right_input(b)))
