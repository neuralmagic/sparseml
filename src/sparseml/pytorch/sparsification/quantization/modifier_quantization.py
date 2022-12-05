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


import logging
import warnings
from itertools import cycle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.modifier import (
    PyTorchModifierYAML,
    ScheduledModifier,
)
from sparseml.pytorch.sparsification.quantization.helpers import (
    configure_module_bn_wrappers,
    freeze_bn_stats,
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
from sparseml.pytorch.utils import BaseLogger, tensors_module_forward, tensors_to_device
from sparseml.sparsification import SparsificationTypes


__all__ = [
    "QuantizationModifier",
]


_LOGGER = logging.getLogger(__name__)


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
    |       module_type_schemes:
    |           Conv2d:
    |               input_activations:
    |                   num_bits: 8
    |                   symmetric: True
    |       exclude_module_types: ["ReLU"]
    |       disable_quantization_observer_epoch: 2.0
    |       freeze_bn_stats_epoch: 3.0

    :param start_epoch: The epoch to start the modifier at
    :param default_scheme: Default QuantizationScheme to use when enabling quantization
        in a module. May also be a dictionary to be loaded into the QuantizationScheme
        class. A string alias may also be used, supported aliases:
        ['default', 'deepsparse', 'tensorrt'].
        If None, the default scheme (`QuantizationScheme()`) will be used.
        Default is None
    :param submodule_schemes: Specify submodules to target for quantization. Must
        be a dictionary of the submodule name to a quantization scheme
        specification to quantize that submodule with.  Modules not included under
        a named submodule in the dictionary will not be targeted for quantization.
        If set to None, the entire module will be quantized falling back to the default
        scheme. Default is None
    :param module_type_schemes: Specify how to quantize specific module types. Must
        be a dictionary of the module type name to a quantization scheme
        specification to quantize that module type with. Default is None
    :param exclude_module_types: optional list of module class names
        to not quantize. Default is None
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param freeze_bn_stats_epoch: Epoch to stop the tracking of batch norm stats. Leave
        None to not stop tracking batch norm stats during QAT. Default is None
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    :param end_epoch: Disabled, setting to anything other than -1 will raise an
        exception. For compatibility with YAML serialization only.
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        default_scheme: QuantizationSchemeLoadable = None,
        submodule_schemes: Optional[Dict[str, QuantizationSchemeLoadable]] = None,
        module_type_schemes: Optional[Dict[str, QuantizationSchemeLoadable]] = None,
        exclude_module_types: Optional[List[str]] = None,
        disable_quantization_observer_epoch: Optional[float] = None,
        freeze_bn_stats_epoch: Optional[float] = None,
        num_calibration_steps: Optional[int] = None,
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
        self._submodule_schemes = _load_quantization_schemes_dict(
            submodule_schemes, self._default_scheme
        )
        self._module_type_schemes = _load_quantization_schemes_dict(
            module_type_schemes, self._default_scheme
        )
        self._exclude_module_types = exclude_module_types
        self._disable_quantization_observer_epoch = disable_quantization_observer_epoch
        self._freeze_bn_stats_epoch = freeze_bn_stats_epoch

        self._num_calibration_steps = num_calibration_steps
        self._calibration_dataloader = None
        self._calibration_function = None

        self._qat_enabled = False
        self._quantization_observer_disabled = False
        self._bn_stats_frozen = False

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
        :return: Specify submodules to target for quantization. Must
            be a dictionary of the submodule name to a quantization scheme
            specification to quantize that submodule with.  Modules not included under
            a named submodule in the dictionary will not be targeted for quantization.
            If set to None, the entire module will be quantized falling back to the
            default scheme.
        """
        return self._submodule_schemes

    @submodule_schemes.setter
    def submodule_schemes(self, value: Optional[Dict[str, QuantizationSchemeLoadable]]):
        """
        :params value:  Specify submodules to target for quantization. Must
            be a dictionary of the submodule name to a quantization scheme
            specification to quantize that submodule with.  Modules not included under
            a named submodule in the dictionary will not be targeted for quantization.
            If set to None, the entire module will be quantized falling back to the
            default scheme.
        """
        self._submodule_schemes = _load_quantization_schemes_dict(
            value, self._default_scheme
        )

    @ModifierProp()
    def module_type_schemes(self) -> Optional[Dict[str, QuantizationSchemeLoadable]]:
        """
        :return: Default Specify how to quantize specific module types. Must
            be a dictionary of the module type name to a quantization scheme
            specification to quantize that module type with.
        """
        return self._module_type_schemes

    @module_type_schemes.setter
    def module_type_schemes(
        self, value: Optional[Dict[str, QuantizationSchemeLoadable]]
    ):
        """
        :params value: Specify how to quantize specific module types. Must
            be a dictionary of the module type name to a quantization scheme
            specification to quantize that module type with.
        """
        self._module_type_schemes = _load_quantization_schemes_dict(
            value, self._default_scheme
        )

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

    @ModifierProp()
    def disable_quantization_observer_epoch(self) -> Optional[float]:
        """
        :return: Epoch to disable updates to the module
            quantization observers. At this point, quantized weights and zero points
            will not be updated. When None, observers never disabled during QAT
        """
        return self._disable_quantization_observer_epoch

    @disable_quantization_observer_epoch.setter
    def disable_quantization_observer_epoch(self, value: Optional[float]):
        """
        :params value: Epoch to disable updates to the module
            quantization observers. At this point, quantized weights and zero points
            will not be updated. Set None to not disable observers during QAT
        """
        self._disable_quantization_observer_epoch = value
        self._validate_params()

    @ModifierProp()
    def freeze_bn_stats_epoch(self) -> Optional[float]:
        """
        :return: Epoch to stop the tracking of batch norm stats. When
            None, batch norm stats are track for all of training
        """
        return self._freeze_bn_stats_epoch

    @freeze_bn_stats_epoch.setter
    def freeze_bn_stats_epoch(self, value: Optional[float]):
        """
        :params value: Epoch to stop the tracking of batch norm stats. Set
            None to not stop tracking batch norm stats during QAT
        """
        self._freeze_bn_stats_epoch = value
        self._validate_params()

    @ModifierProp()
    def num_calibration_steps(self) -> Optional[int]:
        """
        :return: Number of steps to run post training calibration for.
            When None, the entire calibration_dataloader is used
        """
        return self._num_calibration_steps

    @num_calibration_steps.setter
    def num_calibration_steps(self, value: Optional[int]):
        """
        :params value: Number of steps to run post training calibration for.
            When None, the entire calibration_dataloader is used
        """
        self._num_calibration_steps = value

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        calibration_dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
        calibration_function: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Grab the module / submodule to perform QAT on

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param calibration_dataloader: optional dataloader for running post training
            quantization with the given model. if present, calibration will be run
            immediately after quantization is enabled
        :param calibration_function: An Optional callable to use for
            calibration of module parameters post training. Should be able to
            accept a batch of inputs along with a module.
            Example: func(batch, module), Defaults to tensors_module_forward
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)

        self._calibration_dataloader = calibration_dataloader
        self._calibration_function = calibration_function

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
        pending |= self._freeze_bn_stats_update_ready(epoch)
        pending |= self._disable_quantization_observer_update_ready(epoch)

        return pending

    def advance_epochs(self, ref_start_epoch: float = None):
        """
        Advance epoch attributes given a reference start epoch

        :param ref_start_epoch: the reference, i.e. new, start epoch
        """
        if ref_start_epoch is None:
            return

        super().advance_epochs(ref_start_epoch=ref_start_epoch)

        if self._disable_quantization_observer_epoch is not None:
            self._disable_quantization_observer_epoch = (
                max(0.0, self._disable_quantization_observer_epoch) + ref_start_epoch
            )

        if self._freeze_bn_stats_epoch is not None:
            self._freeze_bn_stats_epoch = (
                max(0.0, self._freeze_bn_stats_epoch) + ref_start_epoch
            )
        self._validate_params()

    def _check_quantization_update(
        self, module: Module, epoch: float, steps_per_epoch: int
    ):
        if self.start_pending(epoch, steps_per_epoch) and not self._qat_enabled:
            self._enable_module_qat(module)

        if self._disable_quantization_observer_update_ready(epoch):
            module.apply(torch.quantization.disable_observer)
            self._quantization_observer_disabled = True

        if self._freeze_bn_stats_update_ready(epoch):
            module.apply(freeze_bn_stats)
            self._bn_stats_frozen = True

    def _disable_quantization_observer_update_ready(self, epoch: float) -> bool:
        return (
            self._disable_quantization_observer_epoch is not None
            and epoch >= self._disable_quantization_observer_epoch
            and not self._quantization_observer_disabled
        )

    def _freeze_bn_stats_update_ready(self, epoch: float) -> bool:
        return (
            self._freeze_bn_stats_epoch is not None
            and epoch >= self._freeze_bn_stats_epoch
            and not self._bn_stats_frozen
        )

    def _enable_module_qat(self, module: Module):
        # fuse conv-bn-relu blocks prior to quantization emulation
        fuse_module_conv_bn_relus(module, inplace=True)

        # add quantization_schemes to target submodules
        set_quantization_schemes(
            module,
            default_scheme=self._default_scheme,
            submodule_schemes=self._submodule_schemes,
            module_type_schemes=self._module_type_schemes,
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

        self._calibrate_if_possible(module)

    def _calibrate_if_possible(self, module):
        if self.num_calibration_steps == 0 and self._calibration_dataloader:
            warnings.warn(
                f"num_calibration_steps is {self.num_calibration_steps}."
                f"Calibration data loader will not be used."
            )
        elif self.num_calibration_steps and not self._calibration_dataloader:
            raise ValueError(
                f"num_calibration_steps is {self.num_calibration_steps}. "
                "Calibration data loader is not set. Pass a "
                "calibration_data_loader with initialize(...) method."
            )

        elif not self._calibration_dataloader or not self._qat_enabled:
            return

        elif self._calibration_dataloader:
            self._calibrate(module)

    def _calibrate(self, module):
        _LOGGER.info("Running quantization calibration using calibration_dataloader")

        module_training = module.training
        module.eval()

        forward_fn: Callable = (
            self._calibration_function
            if self._calibration_function
            else tensors_module_forward
        )

        model_device = next(module.parameters()).device
        _dataloader = (
            self._calibration_dataloader
            if self.num_calibration_steps is None
            else cycle(self._calibration_dataloader)
        )

        for batch_idx, batch in enumerate(_dataloader):
            if self.num_calibration_steps and batch_idx >= self.num_calibration_steps:
                break
            batch = tensors_to_device(batch, model_device)
            with torch.no_grad():
                forward_fn(batch, module=module)

        if module_training:
            module.train()

    def _validate_params(self):
        self.validate_schedule()
        if (
            self._disable_quantization_observer_epoch is not None
            and self._disable_quantization_observer_epoch < self._start_epoch
        ):
            raise ValueError(
                f"disable_quantization_observer_epoch may not be greater than "
                f"start_epoch for QuantizationModifier, received: "
                f"{self._disable_quantization_observer_epoch} with start_epoch "
                f"{self._start_epoch}"
            )

        if (
            self._freeze_bn_stats_epoch is not None
            and self._freeze_bn_stats_epoch < self._start_epoch
        ):
            raise ValueError(
                "freeze_bn_stats_epoch may not be greater than start_epoch"
                " for QuantizationModifier, received: {} with start_epoch {}".format(
                    self._freeze_bn_stats_epoch, self._start_epoch
                )
            )


class _QuantizationSchemesDict(dict):
    # wrapper class for dict to override the __str__ method for yaml serialization

    def __str__(self):
        return str({submodule: scheme.dict() for submodule, scheme in self.items()})


def _load_quantization_schemes_dict(
    schemes_dict: Optional[Dict[str, QuantizationSchemeLoadable]],
    default_scheme: QuantizationScheme,
) -> Optional[Dict[str, QuantizationScheme]]:
    if schemes_dict is None:
        return None
    return _QuantizationSchemesDict(
        {
            submodule: QuantizationScheme.load(scheme, default=default_scheme)
            for submodule, scheme in schemes_dict.items()
        }
    )
