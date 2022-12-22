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
import math
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
from sparseml.pytorch.sparsification.quantization.quantization_scheme import (
    QuantizationScheme,
    QuantizationSchemeLoadable,
)
from sparseml.pytorch.sparsification.quantization.quantize import (
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
    After the start epoch, the specified module(s) forward pass will emulate
    quantized execution and the modifier will be enabled until training is completed.

    | Sample yaml:
    |   !QuantizationModifier
    |       start_epoch: 0.0
    |       scheme:
    |           input_activations:
    |               num_bits: 8
    |               symmetric: False
    |           weights:
    |               num_bits: 8
    |               symmetric: True
    |       scheme_overrides:
    |           feature_extractor: "default"
    |           classifier:
    |               input_activations:
    |                   num_bits: 8
    |                   symmetric: False
    |               weights: null
    |           Conv2d:
    |               input_activations:
    |                   num_bits: 8
    |                   symmetric: True
    |       ignore: ["ReLU", "input"]
    |       disable_quantization_observer_epoch: 2.0
    |       freeze_bn_stats_epoch: 3.0
    |       model_fuse_fn_name: 'fuse_module'
    |       strict: True

    :param start_epoch: The epoch to start the modifier at
    :param scheme: Default QuantizationScheme to use when enabling quantization
        in a module. May also be a dictionary to be loaded into the QuantizationScheme
        class. A string alias may also be used, supported aliases:
        ['default', 'deepsparse', 'tensorrt'].
        If None, the default scheme (`QuantizationScheme()`) will be used.
        Default is None
    :param scheme_overrides: optional mapping of module type names or submodule type
        names to quantization schemes to override them with. If a scheme is mapped to
        'default', then it will use the scheme set in the modifier scheme property
    :param ignore: optional list of module class names or submodule names
        to not quantize. Default is None
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param freeze_bn_stats_epoch: Epoch to stop the tracking of batch norm stats. Leave
        None to not stop tracking batch norm stats during QAT. Default is None
    :param model_fuse_fn_name: Name of model function to fuse the model in place prior
        to performing QAT.  Set as None or 'no_fuse' to skip module fusing. Set as
         'conv_bv_relus' to use `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
        Default is None
    :param model_fuse_fn_kwargs: dictionary of keyword argument values to be passed
        to the model fusing function
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    :param strict: if True, will raise an error if any module types or submodules in
        scheme_overrides or ignore are not found in a given module. Default True
    :param end_epoch: Disabled, setting to anything other than -1 will raise an
        exception. For compatibility with YAML serialization only.
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        scheme: QuantizationSchemeLoadable = None,
        scheme_overrides: Optional[Dict[str, QuantizationSchemeLoadable]] = None,
        ignore: Optional[List[str]] = None,
        disable_quantization_observer_epoch: Optional[float] = None,
        freeze_bn_stats_epoch: Optional[float] = None,
        model_fuse_fn_name: Optional[str] = None,
        model_fuse_fn_kwargs: Optional[Dict[str, Any]] = None,
        num_calibration_steps: Optional[int] = None,
        strict: bool = True,
        end_epoch: float = -1.0,
    ):
        raise_if_torch_quantization_not_available()
        if end_epoch != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1. Given {}".format(end_epoch)
            )
        super().__init__(start_epoch=start_epoch, end_epoch=-1.0, end_comparator=-1)

        self._scheme = QuantizationScheme.load(scheme)
        self._scheme_overrides = _load_quantization_schemes_dict(
            scheme_overrides, self._scheme
        )
        self._ignore = ignore or []
        self._disable_quantization_observer_epoch = disable_quantization_observer_epoch
        self._freeze_bn_stats_epoch = freeze_bn_stats_epoch

        self._num_calibration_steps = num_calibration_steps
        self._calibration_dataloader = None
        self._calibration_function = None

        self._model_fuse_fn_name = model_fuse_fn_name
        self._model_fuse_fn_kwargs = model_fuse_fn_kwargs or {}
        if (
            isinstance(self._model_fuse_fn_name, str)
            and self._model_fuse_fn_name.lower() == "none"
        ):
            self._model_fuse_fn_name = None

        self._strict = strict

        self._qat_enabled = False
        self._quantization_observer_disabled = False
        self._bn_stats_frozen = False

        self._validate_params()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.quantization, SparsificationTypes.structured]

    @ModifierProp()
    def scheme(self) -> QuantizationSchemeLoadable:
        """
        :return: Default QuantizationScheme to use when enabling quantization
            in a module. returned as a dictionary for serialization purposes
        """
        return self._scheme

    @scheme.setter
    def scheme(self, value: QuantizationSchemeLoadable):
        """
        :params value: Default QuantizationScheme to use when enabling quantization
            in a module. May also be a dictionary to be loaded into the
            QuantizationScheme class. If None, the default scheme
            (`QuantizationScheme()`) will be used
        """
        self._scheme = QuantizationScheme.load(value)

    @ModifierProp()
    def scheme_overrides(self) -> Optional[Dict[str, QuantizationSchemeLoadable]]:
        """
        :return: optional mapping of module type names or submodule type
            names to quantization schemes to override them with. If a scheme is mapped
            to 'default', then it will use the scheme set in the modifier scheme
            property
        """
        return self._scheme_overrides

    @scheme_overrides.setter
    def scheme_overrides(self, value: Optional[Dict[str, QuantizationSchemeLoadable]]):
        """
        :params value: optional mapping of module type names or submodule type
            names to quantization schemes to override them with. If a scheme is mapped
            to 'default', then it will use the scheme set in the modifier scheme
            property
        """
        self._scheme_overrides = _load_quantization_schemes_dict(value, self._scheme)

    @ModifierProp()
    def ignore(self) -> List[str]:
        """
        :return: optional list of module class names or submodule names to not propagate
            quantization schemes to
        """
        return self._ignore

    @ignore.setter
    def ignore(self, value: Optional[List[str]]):
        """
        :params value: optional list of module class names or submodule names
            to not propagate quantization schemes to
        """
        self._ignore = value or []

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

    @ModifierProp()
    def model_fuse_fn_name(self) -> Optional[str]:
        """
        :return: Name of model function to fuse the model in place prior
            to performing QAT. None sets to default function.
            If tensorrt flag is True, default is 'no_fuse', otherwise
            `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
        """
        return self._model_fuse_fn_name

    @model_fuse_fn_name.setter
    def model_fuse_fn_name(self, value: Optional[str]):
        """
        :params value: Name of model function to fuse the model in place prior
            to performing QAT. Set None to use the default function
            `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.  Set as 'no_fuse'
            to skip module fusing.
        """
        self._model_fuse_fn_name = value
        if (
            isinstance(self._model_fuse_fn_name, str)
            and self._model_fuse_fn_name.lower() == "none"
        ):
            self._model_fuse_fn_name = None
        self._validate_params()

    @ModifierProp()
    def model_fuse_fn_kwargs(self) -> Dict[str, Any]:
        """
        :return: Dictionary of keyword arguments to be passed to the
            model fuse function
        """
        return self._model_fuse_fn_kwargs

    @ModifierProp()
    def strict(self) -> bool:
        """
        :return: if True, will raise an error if any module types or submodules in
            scheme_overrides or ignore are not found in the given module
        """
        return self._strict

    @strict.setter
    def strict(self, value: bool):
        """
        :params value: if True, will raise an error if any module types or submodules in
            scheme_overrides or ignore are not found in the given module
        """
        self._strict = value

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

        self._log_quantization(module, epoch, steps_per_epoch)

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
        self._fuse(module)

        # add quantization_schemes to target submodules
        set_quantization_schemes(
            module,
            scheme=self._scheme,
            scheme_overrides=self._scheme_overrides,
            ignore=self._ignore,
            strict=self._strict,
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

    def _fuse(self, module: Module):
        if self.model_fuse_fn_name in [None, "conv_bn_relus"]:
            self._model_fuse_fn_kwargs["inplace"] = True
            fuse_module_conv_bn_relus(module, **self._model_fuse_fn_kwargs)
        elif self.model_fuse_fn_name != "no_fuse":
            module_fuse_fn = getattr(module, self._model_fuse_fn_name, None)
            if module_fuse_fn is None or not callable(module_fuse_fn):
                raise ValueError(
                    "Invalid model_fuse_fn_name. "
                    "Module has no callable function {}".format(
                        self._model_fuse_fn_name
                    )
                )
            module_fuse_fn(**self._model_fuse_fn_kwargs)

    def _calibrate_if_possible(self, module: Module):
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

    def _calibrate(self, module: Module):
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

        all_schemes = [self._scheme] + list(self._scheme_overrides.values())
        if any(scheme.target_hardware == "tensorrt" for scheme in all_schemes) and (
            self._model_fuse_fn_name != "no_fuse"
        ):
            _LOGGER.info(
                "QuantizationModifier - target hardware tensorrt detected - "
                "Disabling model fuse step"
            )
            self._model_fuse_fn_name = "no_fuse"

    def _log_quantization(
        self,
        module: Module,
        epoch: float,
        steps_per_epoch: int,
    ):
        """
        Check whether to log an update for the learning rate of the modifier.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """

        def _log(tag, value):
            self.log_scalar(
                tag=tag,
                value=value,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

        # log layer-wise quantization info
        num_fake_quantizes = 0
        for name, submodule in module.named_modules():
            if not isinstance(submodule, torch.quantization.FakeQuantize):
                continue
            num_fake_quantizes += 1

            qrange = submodule.quant_max - submodule.quant_min + 1
            num_bits = int(math.log2(qrange))

            _log(
                tag=f"QuantizationModifier/{name}/num_bits",
                value=num_bits,
            )

        # log global quantization info
        _log(
            tag="QuantizationModifier/num_fake_quantize_global",
            value=num_fake_quantizes,
        )
        _log(
            tag="QuantizationModifier/bn_stats_frozen",
            value=1.0 if self._bn_stats_frozen else 0.0,
        )
        _log(
            tag="QuantizationModifier/qat_observers_disabled",
            value=1.0 if self._quantization_observer_disabled else 0.0,
        )


class _QuantizationSchemesDict(dict):
    # wrapper class for dict to override the __str__ method for yaml serialization

    def __str__(self):
        return str({submodule: scheme.dict() for submodule, scheme in self.items()})


def _load_quantization_schemes_dict(
    schemes_dict: Optional[Dict[str, QuantizationSchemeLoadable]],
    default_scheme: QuantizationScheme,
) -> Dict[str, QuantizationScheme]:
    if schemes_dict is None:
        return {}
    return _QuantizationSchemesDict(
        {
            submodule: QuantizationScheme.load(scheme, default=default_scheme)
            for submodule, scheme in schemes_dict.items()
        }
    )
