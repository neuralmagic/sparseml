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
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, Field
from torch.nn import Identity, Module

from sparseml.pytorch.sparsification.quantization.constants import (
    FUSED_MODULE_NAMES,
    NON_QUANTIZABLE_MODULE_NAMES,
)
from sparseml.pytorch.sparsification.quantization.helpers import (
    get_observer,
    prepare_embeddings_qat,
)
from sparseml.pytorch.utils import get_layer


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
    "QuantizationSchemeLoadable",
    "convert_module_qat_from_schemes",
    "is_qat_helper_module",
    "is_quantizable_module",
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

"""
Type definition for a type that is valid for loading a QuantizationScheme
using QuantizationScheme.load
"""
QuantizationSchemeLoadable = Union[
    "QuantizationScheme",
    DictQuantizationScheme,
    str,
    None,
]


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

    @classmethod
    def load(
        cls,
        scheme: QuantizationSchemeLoadable,
        default: Optional["QuantizationScheme"] = None,
    ) -> "QuantizationScheme":
        """
        :param scheme: QuantizationScheme, dict representation of scheme,
            or string alias of a scheme to load. Valid strings: ['default']
        :param default: default QuantizationScheme to override 'default' scheme
            with
        :return: constructed QuantizationScheme object from the given scheme;
            if given a dict, returns QuantizationScheme.parse_obj(scheme), string
            input will return the defualt QuantizationScheme if set to 'default'.
        """
        if isinstance(scheme, cls):
            return scheme
        elif scheme is None or scheme == "default":
            # if no default override, defaults to QuantizationScheme()
            return deepcopy(default) or cls()
        elif isinstance(scheme, str):
            raise ValueError(
                f"Unrecognized QuantizationScheme string alias {scheme}. "
                "Valid strings: ['default']"
            )
        elif isinstance(scheme, dict):
            # default to dict
            scheme = {key: _parse_quantization_arg(arg) for key, arg in scheme.items()}
            return cls.parse_obj(scheme)
        else:
            raise ValueError(
                f"Unrecognized type {type(scheme)} for QuantizationScheme.load, "
                "expected one of: [QuantizationScheme, Dict, str, None]"
            )

    def get_qconfig(self) -> "torch.quantization.QConfig":
        """
        :return: QConfig for Modules (output activations used,
            use QuantWrapper for inputs)
        """
        qconfig = _get_qconfig(self.output_activations, self.weights)
        # add reference to this quantization scheme for reference
        qconfig.quantization_scheme = self
        return qconfig

    def get_wrapper_qconfig(self) -> "torch.quantization.QConfig":
        """
        :return: QConfig for QuantWrapper objects (input activations used)
        """
        qconfig = _get_qconfig(self.input_activations, None)
        # add reference to this quantization scheme for reference
        qconfig.quantization_scheme = self
        return qconfig

    def __str__(self) -> str:
        """
        :return: YAML friendly string serialization
        """
        dict_repr = self.dict()
        dict_repr = {
            key: val if val is not None else "null" for key, val in dict_repr.items()
        }
        return str(dict_repr)


def is_qat_helper_module(module: Module) -> bool:
    """
    :param module: module to check
    :return: True if module is an instance of a torch QAT helper class
    """
    return isinstance(
        module,
        (
            torch_quantization.ObserverBase,
            torch_quantization.FakeQuantize,
            torch_quantization.DeQuantStub,
            torch_quantization.QuantStub,
            Identity,
        ),
    )


def is_quantizable_module(
    module: Module,
    exclude_module_types: Optional[List[str]] = None,
) -> bool:
    """
    :param module: module to check
    :param exclude_module_types: string names of modules to not include for
        quantization. Default None
    :return: boolean value if the module is quantizable. Module is considered
        quantizable if its type is not included in exclude_module_types or
        NON_QUANTIZABLE_MODULE_NAMES and
        it either has no module children outside of QAT or is a torch qat fused module
    """
    # considers any non-excluded "leaf level" (no children) submodule
    # to be quantizable as well as torch fused modules

    # add all default excluded module type names
    exclude_module_types = set(exclude_module_types or [])
    exclude_module_types.update(NON_QUANTIZABLE_MODULE_NAMES)

    module_type_name = module.__class__.__name__
    if module_type_name in exclude_module_types:
        return False

    return (
        module_type_name in FUSED_MODULE_NAMES
        or all(
            # no children (leaf modules) evaluate to all([]) - (True)
            is_qat_helper_module(child)
            for child in module.children()
        )
        or isinstance(module, torch_quantization.QuantWrapper)
    )


def set_quantization_schemes(
    model: Module,
    default_scheme: QuantizationScheme,
    submodule_schemes: Optional[Dict[str, QuantizationScheme]] = None,
    exclude_module_types: Optional[List[str]] = None,
):
    """
    Sets an appropriate `quantization_scheme` to targeted quantizable submodules

    :param model: module to attach QuantizationSchemes to
    :param exclude_module_types: string names of modules to not include for
        quantization. Default None
    :param submodule_schemes:
    :param default_scheme: default scheme to add to a target module unless overwritten
        by another scheme
    """

    def _propagate_quantization_scheme(module: Module, scheme: QuantizationScheme):
        for submodule in module.modules():
            if is_quantizable_module(submodule, exclude_module_types):
                submodule.quantization_scheme = scheme

    if submodule_schemes is None:
        # quantize entire model
        _propagate_quantization_scheme(model, default_scheme)
    else:
        for submodule_name, target_scheme in submodule_schemes.items():
            submodule = get_layer(submodule_name, model)
            _propagate_quantization_scheme(submodule, target_scheme)


def set_qconfigs_from_quantization_schemes(module: Module):
    """
    Sets `qconfig` properties to the given module and its submodule
    based on any potentially assigned quantization schemes

    :param module: module to set qconfig properties for
    """
    for submodule in module.modules():
        if not hasattr(submodule, "quantization_scheme"):
            continue
        if isinstance(submodule, torch_quantization.QuantWrapper):
            submodule.qconfig = submodule.quantization_scheme.get_wrapper_qconfig()
            submodule.quant.qconfig = submodule.qconfig
        else:
            submodule.qconfig = submodule.quantization_scheme.get_qconfig()


def add_input_activation_quant_wrappers(module: Module) -> Module:
    """
    Adds QuantWrapper objects to wrap submodules that include quantization
    schemes targeting input activations

    :param module: module to add input activation QuantWrappers for
    :return: the updated module - necessary in case top level module is wrapped
        as in-place modification will not support it
    """
    # check if module targets input activation quantization
    quantize_activations = (
        hasattr(module, "quantization_scheme")
        and (module.quantization_scheme is not None)
        and module.quantization_scheme.input_activations is not None
    )

    if quantize_activations:
        # wrap module with a QuantWrapper and assign it the input activation qconfig
        quantization_scheme = module.quantization_scheme
        module = torch_quantization.QuantWrapper(module)
        module.quantization_scheme = quantization_scheme

        # assumes no nested children of a wrapped block need input activation
        # does not recurse further in this case
    else:
        # recurse to module children
        for name, child in module.named_children():
            setattr(module, name, add_input_activation_quant_wrappers(child))
    return module


def convert_module_qat_from_schemes(module: Module):
    """
    Converts submodules with set quantization_schemes into quantization aware modules
    with FakeQuantize modules in the model

    :param module: module to convert to QAT mode
    """
    # inject necessary QuantWrappers into the module to apply QAT to
    # targeted layer input activations
    module = add_input_activation_quant_wrappers(module)

    # set appropriate qconfig properties in submodules
    set_qconfigs_from_quantization_schemes(module)

    # set modules with proper qconfigs to QAT mode
    mapping = _get_qat_module_mappings()
    torch_quantization.convert(
        module, mapping=mapping, inplace=True, remove_qconfig=False
    )
    torch_quantization.add_observer_(module, non_leaf_module_list=set(mapping.values()))

    # manual pass to convert relevant Embedding layers
    prepare_embeddings_qat(module)
    # re-attach any quantization schemes lost during conversion
    _reattach_quantization_schemes(module)


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
        activation=activation_args.get_observer() if activation_args else Identity,
        weight=weight_args.get_observer() if weight_args else Identity,
    )


def _reattach_quantization_schemes(module: Module):
    # after torch.prepare_qat is called, quantization scheme properties may be lost
    # due to transfer of base module classes to their QAT implementations
    # this function uses the reference to the quantization_scheme in the qconfig
    # to potentially re-attach the scheme
    for submodule in module.modules():
        qconfig = getattr(submodule, "qconfig", None)
        if not qconfig or hasattr(submodule, "quantization_scheme"):
            # no qconfig, or scheme already set
            continue
        quantization_scheme = getattr(qconfig, "quantization_scheme", None)
        if not quantization_scheme:
            continue
        submodule.quantization_scheme = quantization_scheme


def _get_qat_module_mappings() -> Dict[Module, Module]:
    mappings = torch_quantization.quantization_mappings
    if not hasattr(mappings, "get_default_qat_module_mappings"):
        # legacy
        return mappings.get_qat_module_mappings()
    # latest
    return mappings.get_default_qat_module_mappings()


def _parse_quantization_arg(arg: Any):
    if arg == "None":
        return None
    return arg
