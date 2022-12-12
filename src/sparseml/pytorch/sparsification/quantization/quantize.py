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
from typing import Dict, List, Optional

from torch.nn import Identity, Module

from sparseml.pytorch.sparsification.quantization.constants import (
    FUSED_MODULE_NAMES,
    NON_QUANTIZABLE_MODULE_NAMES,
)
from sparseml.pytorch.sparsification.quantization.helpers import (
    QATWrapper,
    configure_module_default_qconfigs,
    prepare_embeddings_qat,
)
from sparseml.pytorch.sparsification.quantization.quantization_scheme import (
    QuantizationScheme,
)
from sparseml.pytorch.utils import get_layer


try:
    from torch import quantization as torch_quantization
    from torch.nn import intrinsic as torch_intrinsic
except Exception:
    torch_quantization = None
    torch_intrinsic = None


__all__ = [
    "convert_module_qat_from_schemes",
    "is_qat_helper_module",
    "is_quantizable_module",
    "set_quantization_schemes",
    "set_qconfigs_from_quantization_schemes",
    "add_input_activation_quant_wrappers",
    "raise_if_torch_quantization_not_available",
]


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
    module_type_schemes: Optional[Dict[str, QuantizationScheme]] = None,
    exclude_module_types: Optional[List[str]] = None,
):
    """
    Sets an appropriate `quantization_scheme` to targeted quantizable submodules

    :param model: module to attach QuantizationSchemes to
    :param exclude_module_types: string names of modules to not include for
        quantization. Default None
    :param submodule_schemes: dictionary of target submodules to their schemes,
        if given, only the target submodules will have quantization schemes set
    :param module_type_schemes: dictionary of module class names to quantization
        schemes to override the default/submodule target scheme with for the associated
        class
    :param default_scheme: default scheme to add to a target module unless overwritten
        by another scheme
    """
    module_type_schemes = module_type_schemes or {}
    # keep mapping of targets for QATWrapper to inject later so module is not modified
    # during iteration
    wrap_qat_targets = {}  # type: Dict[str, QuantizationScheme]

    def _propagate_quantization_scheme(
        module: Module,
        scheme: QuantizationScheme,
        module_name: str = "",
    ):
        for submodule_name, submodule in module.named_modules():
            if module_name:
                submodule_name = f"{module_name}.{submodule_name}"

            is_scheme_override = submodule.__class__.__name__ in module_type_schemes
            submodule_scheme = (
                scheme
                if not is_scheme_override
                else module_type_schemes[submodule.__class__.__name__]
            )

            if getattr(submodule, "wrap_qat", False):
                # wrap_qat overrides default scheme behavior
                wrap_qat_targets[submodule_name] = submodule_scheme
            elif is_scheme_override or is_quantizable_module(
                submodule, exclude_module_types
            ):
                submodule.quantization_scheme = submodule_scheme

    if submodule_schemes is None:
        # quantize entire model
        _propagate_quantization_scheme(model, default_scheme)
    else:
        for target_name, target_scheme in submodule_schemes.items():
            target_submodule = get_layer(target_name, model)
            _propagate_quantization_scheme(target_submodule, target_scheme)

    for wraped_module_name, scheme in wrap_qat_targets.items():
        _inject_qat_wrapper(model, wraped_module_name, scheme)


def set_qconfigs_from_quantization_schemes(module: Module):
    """
    Sets `qconfig` properties to the given module and its submodule
    based on any potentially assigned quantization schemes

    :param module: module to set qconfig properties for
    """
    for submodule in module.modules():
        if not hasattr(submodule, "quantization_scheme"):
            continue
        # potentially re-load if scheme is set as dict or str
        quantization_scheme = QuantizationScheme.load(submodule.quantization_scheme)
        if isinstance(submodule, torch_quantization.QuantWrapper):
            submodule.qconfig = quantization_scheme.get_wrapper_qconfig()
            submodule.quant.qconfig = submodule.qconfig
        else:
            submodule.qconfig = quantization_scheme.get_qconfig()


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

    # override any qconfigs set in `configure_qconfigs` function
    configure_module_default_qconfigs(module)

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


def _inject_qat_wrapper(
    root_module: Module,
    target_submodule_name: str,
    quantization_scheme: QuantizationScheme,
):
    submodule_name_parts = target_submodule_name.split(".")
    parent_name = ".".join(submodule_name_parts[:-1])

    parent_module = get_layer(parent_name, root_module)
    target_module = getattr(parent_module, submodule_name_parts[-1])

    wrapped_target_module = QATWrapper.from_module(target_module, quantization_scheme)
    setattr(parent_module, submodule_name_parts[-1], wrapped_target_module)


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
