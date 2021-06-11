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
Modules to assist with quantizing PyTorch models
"""

import torch
from torch import Tensor
from torch.nn import Module


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


__all__ = ["QATMatMul"]


class QATMatMul(Module):
    """
    Module to calibrate the inputs and outputs of a matrix multiplication during QAT.
    Can be used as a submodule to a module in place of functional torch.matmul calls.

    Before QAT is enabled, calling QATMatMul().forward(x, y) will be equivalent to
    torch.matmul(x, y)
    """

    def __init__(self):
        if torch_quantization is None:
            raise RuntimeError(
                "Unable to import package torch.quantization. "
                "Try upgrading your PyTorch version."
            )

        super().__init__()

        self.quant_1 = torch_quantization.QuantStub()
        self.quant_2 = torch_quantization.QuantStub()
        self.quant_3 = torch_quantization.QuantStub()  # calibrate matmul outputs
        self.dequant = torch_quantization.DeQuantStub()  # to FP32 if model converted

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        :param x: first matrix for multiplication
        :param y: second matrix for multiplication
        :return: matrix multiplication of x*y. If QAT is enabled, input and output
            ranges will be calibrated
        """
        x = self.quant_1(x)
        y = self.quant_2(y)

        prod = torch.matmul(x, y)
        prod = self.quant_3(prod)
        return self.dequant(prod)
