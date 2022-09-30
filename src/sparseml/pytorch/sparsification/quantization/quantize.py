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
Tooling for applying quantization to pytorch modules via
structured configurations
"""

from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, Field

from sparseml.pytorch.sparsification.quantization.helpers import compute_range


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


__all__ = [
    "QuantizationArgs",
    "QuantizationScheme",
]


class QuantizationArgs(BaseModel):
    """
    Class representing user facing arguments to define quantization Observers of
    activations or weights in a network
    """

    num_bits: int = Field(default=8, help="number of bits to target for quantization")
    symmetric: bool = Field(
        default=False, help="set True to use symmetric quantization. Default False"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        help=(
            "optional dict of kwargs to be passed directly to torch quantization "
            "Observers constructor excluding quantization range or symmetry"
        ),
    )

    @classmethod
    def default_activation_args(cls):
        """
        :return: default 8 bits asymmetric settings
        """
        return cls(num_bits=8, symmetric=False)

    @classmethod
    def default_weight_args(cls):
        """
        :return: default 8 bits symmetric settings
        """
        return cls(num_bits=8, symmetric=True)

    def get_observer(self) -> "torch.quantization.FakeQuantize":
        """
        :return: torch quantization FakeQuantize built based on these QuantizationArgs
        """
        qscheme = (
            torch.per_tensor_symmetric if self.symmetric else torch.per_tensor_affine
        )
        target_dtype = torch.qint8
        quant_min, quant_max = compute_range(target_dtype, self.num_bits)
        return torch_quantization.FakeQuantize.with_args(
            observer=torch_quantization.MovingAverageMinMaxObserver,
            quant_min=quant_min,
            quant_max=quant_max,
            dtype=target_dtype,
            qscheme=qscheme,
            **self.kwargs,
        )


class QuantizationScheme(BaseModel):
    """
    Class composed of QuantizationArgs to build QConfig and QuantWrapper objects for
    quantizing models. Provides a simple user interface for defining how inputs,
    weights, and outputs should be quantized
    """

    input_activations: Optional[QuantizationArgs] = Field(
        default_factory=QuantizationArgs.default_activation_args,
        help=(
            "target quantization setting for input activations. Set to None to "
            "not quantize input activations. Default is 8 bits asymmetric"
        ),
    )
    weights: Optional[QuantizationArgs] = Field(
        default_factory=QuantizationArgs.default_weight_args,
        help=(
            "target quantization setting for model weights. Set to None to "
            "not quantize weights. Default is 8 bits symmetric"
        ),
    )
    output_activations: Optional[QuantizationArgs] = Field(
        default=None,
        help=(
            "target quantization setting for output activations. Set to None to "
            "not quantize output activations. Default is None"
        ),
    )

    def get_qconfig(self) -> "torch.quantization.QConfig":
        """
        :return: QConfig for Modules (output activations used,
            use QuantWrapper for inputs)
        """
        return _get_qconfig(self.output_activations, self.weights)

    def get_wrapper_qconfig(self) -> "torch.quantization.QConfig":
        """
        :return: QConfig for QuantWrapper objets (input activations used)
        """
        return _get_qconfig(self.input_activations, self.weights)


def _get_qconfig(
    activation_args: Optional[QuantizationArgs], weight_args: Optional[QuantizationArgs]
) -> "torch.quantization.QConfig":
    return torch_quantization.QConfig(
        activation=activation_args.get_observer() if activation_args else None,
        weight=weight_args.get_observer() if weight_args else None,
    )
