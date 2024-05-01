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

import logging
from typing import Any, Dict, Optional

import torch
from torch.nn import Module

from sparseml.core import Event, EventType, State
from sparseml.modifiers.quantization.base import QuantizationModifier
from sparseml.modifiers.quantization.modification import modify_model
from sparseml.modifiers.quantization.utils.helpers import (
    configure_module_bn_wrappers,
    freeze_bn_stats,
    fuse_module_conv_bn_relus,
)
from sparseml.modifiers.quantization.utils.quantization_scheme import (
    QuantizationScheme,
    QuantizationSchemeLoadable,
)
from sparseml.modifiers.quantization.utils.quantize import (
    convert_module_qat_from_schemes,
    raise_if_torch_quantization_not_available,
    set_quantization_schemes,
)
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward
from sparseml.utils.fsdp.context import summon_full_params_context


_LOGGER = logging.getLogger(__name__)


class QuantizationModifierPyTorch(QuantizationModifier):
    """
    Pytorch-specific implementation of quantization modifier

    :param scheme: Default QuantizationScheme to use when enabling quantization
        in a module. May also be a dictionary to be loaded into the QuantizationScheme
        class. A string alias may also be used, supported aliases:
        ['default', 'deepsparse', 'tensorrt'].
        If None, the default scheme (`QuantizationScheme()`) will be used.
        Default is None
    :param scheme_overrides: optional mapping of module type names or submodule type
        names to quantization schemes to override them with. If a scheme is mapped to
        'default', then it will use the scheme set in the modifier scheme property
    """

    scheme: Optional[QuantizationSchemeLoadable] = None
    scheme_overrides: Optional[Dict[str, QuantizationSchemeLoadable]] = None
    calibration_dataloader_: Any = None
    calibration_function_: Any = None
    qat_enabled_: bool = False
    quantization_observer_disabled_: bool = False
    bn_stats_frozen_: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheme = QuantizationScheme.load(self.scheme)
        self.scheme_overrides = _load_quantization_schemes_dict(
            self.scheme_overrides, self.scheme
        )

    def on_initialize_structure(self, state: State, **kwargs):
        module = state.model.model
        # before the structure is modified to support quantization,
        # we need to potentially modify the model architecture
        module = modify_model(module)
        self._enable_module_qat(module)
        state.model.model.apply(torch.quantization.disable_observer)

    def on_initialize(self, state: State, **kwargs) -> bool:
        raise_if_torch_quantization_not_available()
        module = state.model.model
        module = modify_model(module)
        if self.end and self.end != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1 or None. Given {}".format(self.end)
            )

        self.calibration_dataloader_ = state.data.calib
        module = state.model.model

        if self.calculate_start() == -1:  # one-shot
            self._enable_module_qat(module)
            self._calibrate_if_possible(module)
            self._disable_quantization_observer(module)

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if self.post_oneshot_calibration:
            state.model.model.apply(torch.quantization.enable_observer)
            self._calibrate_if_possible(state.model.model)
        self._disable_quantization_observer(state.model.model)
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        if not self.qat_enabled_:
            self._enable_module_qat(state.model.model)

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.BATCH_START:
            if self.check_should_freeze_bn_stats(event):
                self._freeze_bn_stats(state.model.model)
            if self.check_should_disable_observer(event):
                self._disable_quantization_observer(state.model.model)

    def on_end(self, state: State, event: Event, **kwargs):
        self._disable_quantization_observer(state.model.model)

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def _freeze_bn_stats(self, model: Module):
        model.apply(freeze_bn_stats)
        self.bn_stats_frozen_ = True

    def _disable_quantization_observer(self, model: Module):
        model.apply(torch.quantization.disable_observer)
        self.quantization_observer_disabled_ = True

    def _enable_module_qat(self, module: Module):
        module.apply(torch.quantization.enable_observer)

        if not self.qat_enabled_:
            with summon_full_params_context(module):
                # fuse conv-bn-relu blocks prior to quantization emulation
                self._fuse(module)

                # add quantization_schemes to target submodules
                set_quantization_schemes(
                    module,
                    scheme=self.scheme,
                    scheme_overrides=self.scheme_overrides,
                    ignore=self.ignore,
                    strict=self.strict,
                )

                # fix for freezing batchnorm statistics when not fusing BN with convs.
                # pytorch only supports freezing batchnorm statistics for fused modules.
                # this fix wraps BN modules adding with a new module class that supports
                # methods related to freezing/unfreezing BN statistics.
                configure_module_bn_wrappers(module)

                # convert target qconfig layers to QAT modules with FakeQuantize
                convert_module_qat_from_schemes(module)

        self.qat_enabled_ = True

    def _fuse(self, module: Module):
        if self.model_fuse_fn_name in [None, "conv_bn_relus"]:
            self.model_fuse_fn_kwargs["inplace"] = True
            fuse_module_conv_bn_relus(module, **self.model_fuse_fn_kwargs)
        elif self.model_fuse_fn_name != "no_fuse":
            module_fuse_fn = getattr(module, self.model_fuse_fn_name, None)
            if module_fuse_fn is None or not callable(module_fuse_fn):
                raise ValueError(
                    "Invalid model_fuse_fn_name. "
                    "Module has no callable function {}".format(self.model_fuse_fn_name)
                )
            module_fuse_fn(**self.model_fuse_fn_kwargs)

    def _calibrate_if_possible(self, module: Module):
        if self.num_calibration_steps == 0 and self.calibration_dataloader_:
            _LOGGER.warning(
                f"num_calibration_steps is {self.num_calibration_steps}."
                f"Calibration data loader will not be used."
            )
        elif self.num_calibration_steps and not self.calibration_dataloader_:
            raise ValueError(
                f"num_calibration_steps is {self.num_calibration_steps}. "
                "Calibration data loader is not set. Pass a "
                "calibration_data_loader with initialize(...) method."
            )

        elif not self.calibration_dataloader_:
            return

        self._calibrate(module)

    def _calibrate(self, module: Module):
        class_name = self.__class__.__name__.replace("PyTorch", "")
        _LOGGER.info(
            f"Running {class_name} calibration with "
            f"{len(self.calibration_dataloader_)} samples..."
        )

        module_training = module.training
        module.eval()

        run_calibration_forward(
            module,
            self.calibration_dataloader_,
            self.num_calibration_steps,
            self.calibration_function_,
        )

        if module_training:
            module.train()
        else:
            self._disable_quantization_observer(module)


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
