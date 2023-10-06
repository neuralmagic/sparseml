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

import torch
import torch.nn as nn


class WeightFakeQuantizer(nn.Module):
    def __init__(self, layer):
        super(WeightFakeQuantizer, self).__init__()
        if not isinstance(layer, nn.qat.Linear):
            raise ValueError("WeightFakeQuantizer expects Linear weight only")
        [
            setattr(self, attr, getattr(layer.weight_fake_quant, attr))
            for attr in ["scale", "zero_point", "dtype", "qscheme"]
        ]

    def quantize(self, w: torch.Tensor):
        if self.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            q = torch.quantize_per_tensor(w, self.scale, self.zero_point, self.dtype)
        else:
            q = torch.quantize_per_channel(
                w, self.scale, self.zero_point, 0, self.dtype
            )
        return torch.dequantize(q)


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
