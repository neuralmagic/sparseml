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

from typing import Any, Dict, Optional, Union

import torch
from pydantic import BaseModel, Field
from torch.nn import Module

from sparseml.pytorch.sparsification.quantization.helpers import (
    get_observer,
    is_quantizable_module,
    prepare_embeddings_qat,
)


try:
    from torch import quantization as torch_quantization
    from torch.nn import intrinsic as torch_intrinsic
except Exception:
    torch_quantization = None
    torch_intrinsic = None


__all__ = [
    "DictQuantizationArgs",
    "DictQuantizationScheme",
    "QuantizationArgs",
    "QuantizationScheme",
    "set_quantization_schemes",
    "set_qconfigs_from_quantization_schemes",
    "add_input_activation_quant_wrappers",
    "raise_if_torch_quantization_not_available",
]


"""
Type definition aliases for defining QuantizationArgs and QuantizationScheme
as dictionaries for YAML serialization
"""
DictQuantizationArgs = Dict[str, Union[int, bool, Dict[str, Any]]]
DictQuantizationScheme = Dict[str, DictQuantizationArgs]


class QuantizationArgs(BaseModel):
    """
    Class representing user facing arguments to define quantization Observers of
    activations or weights in a network
    """

    num_bits: int = Field(
        default=8, description="number of bits to target for quantization"
    )
    symmetric: bool = Field(
        default=False,
        description="set True to use symmetric quantization. Default False",
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
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
        return get_observer(
            symmetric=self.symmetric,
            dtype=torch.qint8,
            bits=self.num_bits,
            reduce_range=self.kwargs.get("reduce_range", False),
            qconfig_kwargs=self.kwargs,
        )


class QuantizationScheme(BaseModel):
    """
    Class composed of QuantizationArgs to build QConfig and QuantWrapper objects for
    quantizing models. Provides a simple user interface for defining how inputs,
    weights, and outputs should be quantized
    """

    def __init__(self, *args, **kwargs):
        # support for loading from yaml str
        args = [arg if arg != "null" else None for arg in args]
        for key, val in kwargs.items():
            if val == "null":
                kwargs[key] = None
        super().__init__(*args, **kwargs)

    input_activations: Optional[QuantizationArgs] = Field(
        default_factory=QuantizationArgs.default_activation_args,
        description=(
            "target quantization setting for input activations. Set to None to "
            "not quantize input activations. Default is 8 bits asymmetric"
        ),
    )
    weights: Optional[QuantizationArgs] = Field(
        default_factory=QuantizationArgs.default_weight_args,
        description=(
            "target quantization setting for model weights. Set to None to "
            "not quantize weights. Default is 8 bits symmetric"
        ),
    )
    output_activations: Optional[QuantizationArgs] = Field(
        default=None,
        description=(
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
        :return: QConfig for QuantWrapper objects (input activations used)
        """
        return _get_qconfig(self.input_activations, self.weights)

    def __str__(self) -> str:
        """
        :return: YAML friendly string serialization
        """
        dict_repr = self.dict()
        dict_repr = {
            key: val if val is not None else "null" for key, val in dict_repr.items()
        }
        return str(dict_repr)


def set_quantization_schemes(module: Module, default_scheme: QuantizationScheme):
    """
    Sets an appropriate `quantization_scheme` porperty to targeted sections of the
    given module based on inputs

    :param module: module to attach QuantizationSchemes to
    :param default_scheme: default scheme to add to a target module unless overwritten
        by another scheme
    """
    module.quantization_scheme = default_scheme


def set_qconfigs_from_quantization_schemes(module: Module):
    """
    Sets `qconfig` properties to the given module and its submodule
    based on any potentially assigned quantization schemes

    :param module: module to set qconfig properties for
    """
    for submodule in module.modules():
        if not hasattr(submodule, "quantization_scheme"):
            continue
        submodule.qconfig = submodule.quantization_scheme.get_qconfig()


def add_input_activation_quant_wrappers(module: Module) -> Module:
    """
    Adds QuantWrapper objects to wrap submodules that include quantization
    schemes targeting input activations

    :param module: module to add input activation QuantWrappers for
    :return: the updated module - necessary in case top level module is wrapped
        as in-place modification will not support it
    """
    # check if module type is appropriate for quantization
    is_quantizable = is_quantizable_module(module)

    # check if module targets input activation quantization
    quantize_activations = (
        hasattr(module, "quantization_scheme")
        and (module.quantization_scheme is not None)
        and module.quantization_scheme.input_activations is not None
    )

    if is_quantizable and quantize_activations:
        # wrap module with a QuantWrapper and assign it the input activation qconfig
        wrapper_qconfig = module.quantization_scheme.get_wrapper_qconfig()
        module = torch_quantization.QuantWrapper(module)
        module.qconfig = wrapper_qconfig

        # assumes no nested children of a wrapped block need input activation
        # does not recurse further in this case
    else:
        # recurse to module children
        for name, child in module.named_children():
            setattr(module, name, add_input_activation_quant_wrappers(child))
    return module


def prepare_module_qat(module: Module):
    """
    Converts submodules with set qconfigs into quantization aware modules
    with FakeQuantize modules in the model

    :param module: module to convert to QAT mode
    """
    # set modules with proper qconfigs to QAT mode
    torch_quantization.prepare_qat(module, inplace=True)
    # manual pass to convert relevant Embedding layers
    prepare_embeddings_qat(module)


def raise_if_torch_quantization_not_available():
    """
    :raises: RuntimeError if the installed torch version does not include
        support for quantization aware training
    """
    if torch_quantization is None or torch_intrinsic is None:
        raise RuntimeError(
            "Unable to import package torch.quantization and/or "
            "torch.nn.intrinsic. "
            "Try upgrading your PyTorch version to use the QuantizationModifier."
        )


def _get_qconfig(
    activation_args: Optional[QuantizationArgs], weight_args: Optional[QuantizationArgs]
) -> "torch.quantization.QConfig":
    return torch_quantization.QConfig(
        activation=activation_args.get_observer() if activation_args else None,
        weight=weight_args.get_observer() if weight_args else None,
    )
