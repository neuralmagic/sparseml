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

import torch
from packaging import version
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
    "add_output_activation_observers",
    "raise_if_torch_quantization_not_available",
]


def is_qat_helper_module(module: Module) -> bool:
    """
    :param module: module to check
    :return: True if module is an instance of a torch QAT helper class
    """
    # prefer FakeQuantizeBase which was introduced around torch 1.9
    fake_quantize_class = getattr(
        torch_quantization, "FakeQuantizeBase", torch_quantization.FakeQuantize
    )
    return isinstance(
        module,
        (
            fake_quantize_class,
            torch_quantization.ObserverBase,
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
    scheme: QuantizationScheme,
    scheme_overrides: Optional[Dict[str, QuantizationScheme]] = None,
    ignore: Optional[List[str]] = None,
    strict: bool = True,
):
    """
    Sets an appropriate `quantization_scheme` to targeted quantizable submodules

    :param model: module to attach QuantizationSchemes to
    :param scheme: default scheme to add to a target module unless overwritten
        by another scheme
    :param scheme_overrides: dictionary of module type names or submodule names
        mapped to a quantization scheme to override with. If a submodule matches
        to multiple submodule overrides and/or a module type, module type will
        take the highest priority followed by the longest matched submodule name
    :param ignore: string names of modules type names or submodule names to not include
        for quantization. Default None
    :param strict: if True, will raise an error if any module types or submodules in
        scheme_overrides or ignore are not found in the given module. Default True
    """
    # default to empty dict
    scheme_overrides = scheme_overrides or {}

    if strict:
        _validate_set_module_schemes(model, scheme_overrides, ignore)

    # keep mapping of targets for QATWrapper to inject later so module is not modified
    # during iteration
    wrap_qat_targets = {}  # type: Dict[str, QuantizationScheme]

    for submodule_name, submodule in model.named_modules():
        if ignore and _match_submodule_name_or_type(submodule, submodule_name, ignore):
            # submodule type or graph section set to ignore, skip
            continue

        # override default scheme if necessary
        override_key = _match_submodule_name_or_type(
            submodule, submodule_name, scheme_overrides
        )
        submodule_scheme = (
            scheme if override_key is None else scheme_overrides[override_key]
        )
        is_module_type_override = override_key == submodule.__class__.__name__

        if getattr(submodule, "wrap_qat", False):
            # wrap_qat overrides default scheme behavior
            wrap_qat_targets[submodule_name] = submodule_scheme
        elif is_module_type_override or is_quantizable_module(submodule):
            # is base quantizable module or user specifically targeted module type
            submodule.quantization_scheme = submodule_scheme

    # inject any targeted QATWrappers
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
        and not isinstance(module, torch.nn.quantized.FloatFunctional)
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


def add_output_activation_observers(module: Module):
    """
    implementation of torch.quantization add_observers_ that only adds observers
    according to attached quantization_scheme properties. the existing implementation
    (1.9+) includes its own logic for propagating including overriding set qconfigs
    for certain activations without the ability to disable this behavior

    :param module: module to add output activation observers to
    """
    # adapted from torch/ao/quantization/quantize.py::_add_observer_
    # source: https://github.com/pytorch/pytorch/blob/v1.13.0/torch/ao/quantization/quantize.py#L135  # noqa: E501
    try:
        device = next(module.parameters()).device
    except StopIteration:
        # default to CPU if module has no parameters
        device = "cpu"

    def _needs_observer(target_module: Module):
        # combines logic from multiple places of original implementation which
        # mostly checked for existnace of a qconfig and if the target was a leaf
        # module
        if not hasattr(target_module, "quantization_scheme") or isinstance(
            target_module, torch_quantization.QuantWrapper
        ):
            # submodule not targeted for quantization, already has attached
            # output observer, or is QuantWrapper (quant wrapper delegates to children)
            return False

        if hasattr(target_module, "activation_post_process"):
            # activation post process is set, only mark for potential overriding
            # if it is an identity (this comes up when the property is set for
            # later overriding such as FloatFunctional
            return isinstance(target_module.activation_post_process, Identity)

        for descendent_module in target_module.modules():
            if descendent_module is target_module:
                continue  # skip itself
            descendent_scheme = getattr(descendent_module, "quantization_scheme", None)
            if descendent_scheme is not None and (
                descendent_scheme.output_activations is not None
            ):
                # a descendent of this module targets output activations, return False
                return False
        # module has a quantization scheme and no descendents track output activations
        return True

    def _observer_forward_hook(self, inp, output):
        # reference for output activation observer hook to register
        return self.activation_post_process(output)

    def _add_activation_post_process(target_module: Module):
        # get output observer
        output_observer = submodule.qconfig.activation()
        output_observer.to(device)

        # add an activation post process module
        target_module.add_module("activation_post_process", output_observer)

        # add hook to call observer after output activation has been returned
        handle = target_module.register_forward_hook(_observer_forward_hook)
        target_module._forward_hooks.move_to_end(handle.id, last=False)

    for submodule in module.modules():
        if not _needs_observer(submodule):
            # submodule not targeted for quantization, already has attached
            # output observer, or has a descendent that tracks output activations
            continue

        # extract qconfig and observer from qconfig
        if not hasattr(submodule, "qconfig"):
            # set qconfig from scheme if not already set
            set_qconfigs_from_quantization_schemes(submodule)
        assert hasattr(submodule, "qconfig")

        # create observer, add as child module, and register hook to call
        _add_activation_post_process(submodule)


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
    convert_kwargs = (
        dict(convert_custom_config_dict={})  # do not let torch override any qconfigs
        if version.parse(torch.__version__) >= version.parse("1.8.0")
        else {}
    )
    torch_quantization.convert(
        module,
        mapping=_get_qat_module_mappings(),
        inplace=True,
        remove_qconfig=False,
        **convert_kwargs,
    )
    # re-attach any quantization schemes lost during conversion
    _reattach_quantization_schemes(module)

    # add observers for output activations
    add_output_activation_observers(module)

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


def _match_submodule_name_or_type(
    submodule: Module, submodule_name: str, names_or_types: List[str]
) -> Optional[str]:
    # match preferences:
    #   1. match module type name
    #   2. match the submodule prefix (longest first)
    submodule_match = ""
    for name_or_type in names_or_types:
        if name_or_type == submodule.__class__.__name__:
            # type match, return type name
            return name_or_type
        if submodule_name.startswith(name_or_type) and (
            len(name_or_type) > len(submodule_match)
        ):
            # match to most specific submodule name
            submodule_match = name_or_type
    return submodule_match or None  # return None if no match


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


def _validate_set_module_schemes(
    model: Module,
    scheme_overrides: Optional[Dict[str, QuantizationScheme]] = None,
    ignore: Optional[List[str]] = None,
):
    def _get_unmatched_types_or_names(types_or_names):
        unmatched = []
        for type_or_name in types_or_names:
            matched = False
            for submodule_name, submodule in model.named_modules():
                if submodule_name.startswith(type_or_name) or (
                    submodule.__class__.__name__ == type_or_name
                ):
                    matched = True
                    break
            if not matched:
                unmatched.append(type_or_name)
        return unmatched

    def _build_error_str(property_name, unmatched_values):
        return (
            f"{property_name} contains submodule names or module types "
            "that do not match to any submodules in the model. "
            f"unmatched values: {unmatched_values}"
        )

    unmatched_scheme_overrides = _get_unmatched_types_or_names(scheme_overrides)
    if unmatched_scheme_overrides:
        raise ValueError(
            _build_error_str("scheme_overrides", unmatched_scheme_overrides)
        )

    unmatched_ignore = _get_unmatched_types_or_names(ignore)
    if unmatched_ignore:
        raise ValueError(_build_error_str("ignore", unmatched_ignore))
