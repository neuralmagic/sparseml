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

"""
Set of helper objects that are used to modify
the quantized models
"""

import torch


__all__ = [
    "QuantizableIdentity",
    "QuantizableMatMul",
    "QuantizableBatchMatmul",
    "QATMatMul",
    "QATLinear",
]


class QuantizableIdentity(torch.nn.Module):
    """
    Identity model that is introduced to be used
    together with QuantizableMatMul to allow for
    SparseML quantization scheme
    """

    def forward(self, x):
        return x


class QuantizableMatMul(torch.nn.Module):
    """
    Wrapper around torch.matmul with distinct inputs/output class
    instances that could be quantized through SparseML recipe

    :param left_input_cls: class instance that is used to quantize the left input
    :param right_input_cls: class instance that is used to quantize the right input
    :param output_cls: class instance that is used to quantize the output (optional)
    :return: the output of the matrix multiplication
    """

    def __init__(self, left_input_cls, right_input_cls, output_cls=None):
        super().__init__()
        self.left_input = left_input_cls()
        self.right_input = right_input_cls()
        self.output = output_cls() if output_cls is not None else None

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        out = torch.matmul(self.left_input(a), self.right_input(b))
        if self.output is not None:
            return self.output(out)
        return out


class QuantizableBatchMatmul(QuantizableMatMul):
    """
    Wrapper around torch.bmm with distinct inputs/output class
    instances that could be quantized through SparseML recipe

    :param left_input_cls: class instance that is used to quantize the left input
    :param right_input_cls: class instance that is used to quantize the right input
    :param output_cls: class instance that is used to quantize the output (optional)
    :return: the output of the batch matrix multiplication
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        out = torch.bmm(self.left_input(a), self.right_input(b))
        if self.output is not None:
            return self.output(out)
        return out


class QATMatMul(torch.nn.Module):
    """
    Behaves like normal torch.matmul unless a SparseML QuantizationModifier
    is initialized (Quantization-Aware-Training is invoked)
    """

    def __init__(self):
        super().__init__()

        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 2,
            "input_qconfigs": ["asymmetric", "symmetric"],
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(a, b)


class QATLinear(torch.nn.Module):
    """
    Behaves like normal torch.nn.Linear unless a SparseML QuantizationModifier
    is initialized (Quantization-Aware-Training is invoked)
    When initialized does not quantize inputs. Only weights are quantized
    (inputs may come quantized)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 0,
            "num_outputs": 1,
        }

        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        return self.linear(x)
