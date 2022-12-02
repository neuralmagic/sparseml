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
Modifier for models through quantization aware training.

PyTorch version must support quantization (>=1.2, ONNX export support introduced in 1.7)
"""


from typing import Any, Dict, List, Optional, Type, Union

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.pytorch.sparsification.quantization.helpers import (
    configure_module_bn_wrappers,
    fuse_module_conv_bn_relus,
)
from sparseml.pytorch.sparsification.quantization.legacy_modifier_quantization import (
    QuantizationModifier as LegacyQuantizationModifier,
)
from sparseml.pytorch.sparsification.quantization.quantize import (
    QuantizationScheme,
    QuantizationSchemeLoadable,
    convert_module_qat_from_schemes,
    raise_if_torch_quantization_not_available,
    set_quantization_schemes,
)
from sparseml.pytorch.utils import BaseLogger
from sparseml.sparsification import SparsificationTypes


__all__ = [
    "QuantizationModifier",
]


# do not move, required to be defined before PyTorchModifierYAML decorator
def _select_quantization_modifier(state: Dict[str, Any]) -> Type:
    # if kwargs for the legacy quantization modifier are provided,
    # route YAML loading to that class
    return LegacyQuantizationModifier if "submodules" in state else QuantizationModifier


@PyTorchModifierYAML(swap_class_by_state_fn=_select_quantization_modifier)
class QuantizationModifier(ScheduledModifier):
    """
    Enables quantization aware training (QAT) for a given module or its submodules
    After the start epoch, the specified module(s)' forward pass will emulate
    quantized execution and the modifier will be enabled until training is completed.

    | Sample yaml:
    |   !QuantizationModifier
    |       start_epoch: 0.0
    |       default_scheme:
    |           input_activations:
    |               num_bits: 8
    |               symmetric: False
    |           weights:
    |               num_bits: 8
    |               symmetric: True
    |       submodule_schemes:
    |           feature_extractor: "default"
    |           classifier:
    |               input_activations:
    |                   num_bits: 8
    |                   symmetric: False
    |               weights: null
    |       exclude_module_types: ["ReLU"]

    :param start_epoch: The epoch to start the modifier at
    :param default_scheme: Default QuantizationScheme to use when enabling quantization
        in a module. May also be a dictionary to be loaded into the QuantizationScheme
        class. A string alias may also be used, supported aliases: ['default'].
        If None, the default scheme (`QuantizationScheme()`) will be used.
        Default is None
    :param submodule_schemes: Specify submodules to target for quantization. Must
        be a dictionary of the submodule name to the a quantization scheme
        specification to quantize that submodule with.  Modules not included under
        a named submodule in the dictionary will not be targeted for quantization.
        If set to None, the entire module will be quantized falling back to the default
        scheme. Default is None
    :param exclude_module_types: optional list of module class names
        to not quantize. Default is None
    :param end_epoch: Disabled, setting to anything other than -1 will raise an
        exception. For compatibility with YAML serialization only.
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        default_scheme: QuantizationSchemeLoadable = None,
        submodule_schemes: Optional[Dict[str, QuantizationSchemeLoadable]] = None,
        exclude_module_types: Optional[List[str]] = None,
        end_epoch: float = -1.0,
    ):
        raise_if_torch_quantization_not_available()
        if end_epoch != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1. Given {}".format(end_epoch)
            )
        super().__init__(start_epoch=start_epoch, end_epoch=-1.0, end_comparator=-1)

        self._default_scheme = QuantizationScheme.load(default_scheme)
        self._submodule_schemes = _load_submodule_schemes(
            submodule_schemes, self._default_scheme
        )
        self._exclude_module_types = exclude_module_types

        self._qat_enabled = False

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.quantization, SparsificationTypes.structured]

    @ModifierProp()
    def default_scheme(self) -> QuantizationSchemeLoadable:
        """
        :return: Default QuantizationScheme to use when enabling quantization
            in a module. returned as a dictionary for serialization purposes
        """
        return self._default_scheme

    @default_scheme.setter
    def default_scheme(self, value: QuantizationSchemeLoadable):
        """
        :params value: Default QuantizationScheme to use when enabling quantization
            in a module. May also be a dictionary to be loaded into the
            QuantizationScheme class. If None, the default scheme
            (`QuantizationScheme()`) will be used
        """
        self._default_scheme = QuantizationScheme.load(value)

    @ModifierProp()
    def submodule_schemes(self) -> Optional[Dict[str, QuantizationSchemeLoadable]]:
        """
        :return: Default QuantizationScheme to use when enabling quantization
            in a module. returned as a dictionary for serialization purposes
        """
        return self._submodule_schemes

    @submodule_schemes.setter
    def submodule_schemes(self, value: Optional[Dict[str, QuantizationSchemeLoadable]]):
        """
        :params value: Default QuantizationScheme to use when enabling quantization
            in a module. May also be a dictionary to be loaded into the
            QuantizationScheme class. If None, the default scheme
            (`QuantizationScheme()`) will be used
        """
        self._submodule_schemes = _load_submodule_schemes(value, self._default_scheme)

    @ModifierProp()
    def exclude_module_types(self) -> Optional[List[str]]:
        """
        :return: optional list of module class names to not propagate
            quantization configs to. Default is None
        """
        return self._exclude_module_types

    @exclude_module_types.setter
    def exclude_module_types(self, value: Optional[List[str]]):
        """
        :params value: Default QuantizationScheme to use when enabling quantization
            in a module. May also be a dictionary to be loaded into the
            QuantizationScheme class. If None, the default scheme
            (`QuantizationScheme()`) will be used
        """
        self._exclude_module_types = value

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the module / submodule to perform QAT on

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)

        self._check_quantization_update(module, epoch, steps_per_epoch=0)

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), fuses the model, sets the model quantization config,
        calls torch.quantization.prepare_qat on the model to begin QAT
        If end_pending(), updates the modules layers params to their original
        trainable state.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        self._check_quantization_update(module, epoch, steps_per_epoch)

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        pending = self.start_pending(epoch, steps_per_epoch)

        return pending

    def _check_quantization_update(
        self, module: Module, epoch: float, steps_per_epoch: int
    ):
        if self.start_pending(epoch, steps_per_epoch) and not self._qat_enabled:
            self._enable_module_qat(module)

    def _enable_module_qat(self, module: Module):
        # fuse conv-bn-relu blocks prior to quantization emulation
        fuse_module_conv_bn_relus(module, inplace=True)

        # add quantization_schemes to target submodules
        set_quantization_schemes(
            module,
            default_scheme=self._default_scheme,
            submodule_schemes=self._submodule_schemes,
            exclude_module_types=self._exclude_module_types,
        )

        # fix for freezing batchnorm statistics when not fusing BN with convs.
        # pytorch only supports freezing batchnorm statistics for fused modules.
        # this fix wraps BN modules adding with a new module class that supports
        # methods related to freezing/unfreezing BN statistics.
        configure_module_bn_wrappers(module)

        # convert target qconfig layers to QAT modules with FakeQuantize
        convert_module_qat_from_schemes(module)

        self._qat_enabled = True


class _QuantizationSchemesDict(dict):
    # wrapper class for dict to override the __str__ method for yaml serialization

    def __str__(self):
        return str({submodule: scheme.dict() for submodule, scheme in self.items()})


def _load_submodule_schemes(
    submodule_schemes: Optional[Dict[str, QuantizationSchemeLoadable]],
    default_scheme: QuantizationScheme,
) -> Optional[Dict[str, QuantizationScheme]]:
    if submodule_schemes is None:
        return None
    if not isinstance(submodule_schemes, dict):
        raise ValueError(
            "submodule_schemes must be set to None or a dict. "
            f"Found {type(submodule_schemes)}"
        )
    return _QuantizationSchemesDict(
        {
            submodule: QuantizationScheme.load(scheme, default=default_scheme)
            for submodule, scheme in submodule_schemes.items()
        }
    )
