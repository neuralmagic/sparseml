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
from itertools import cycle
from typing import Any, Callable

import torch
from torch.nn import Module

from sparseml.core import Event, State
from sparseml.modifiers.quantization.base import QuantizationModifier
from sparseml.modifiers.quantization.utils.helpers import (
    configure_module_bn_wrappers,
    fuse_module_conv_bn_relus,
)
from sparseml.modifiers.quantization.utils.quantize import (
    convert_module_qat_from_schemes,
    raise_if_torch_quantization_not_available,
    set_quantization_schemes,
)
from sparseml.pytorch.utils import tensors_module_forward, tensors_to_device


_LOGGER = logging.getLogger(__name__)


class QuantizationModifierPyTorch(QuantizationModifier):
    calibration_dataloader_: Any = None
    calibration_function_: Any = None
    qat_enabled_: bool = False

    def on_initialize(self, state: State, **kwargs) -> bool:
        raise_if_torch_quantization_not_available()
        if self.end and self.end != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1 or None. Given {}".format(self.end)
            )

        self.calibration_dataloader_ = state.data.calib
        module = state.model.model
        device = state.hardware.device
        state.model.model.to(device)
        module = state.model.model
        self._enable_module_qat(module)

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if self.post_oneshot_calibration:
            state.model.model.to(state.hardware.device)
            state.model.model.apply(torch.quantization.enable_observer)
            self._calibrate_if_possible(state.model.model)
        state.model.model.apply(torch.quantization.disable_observer)
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def _enable_module_qat(self, module: Module):
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

        self._calibrate_if_possible(module)

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
        _LOGGER.info("Running quantization calibration using calibration_dataloader")

        module_training = module.training
        module.eval()

        forward_fn: Callable = (
            self.calibration_function_
            if self.calibration_function_
            else tensors_module_forward
        )

        model_device = next(module.parameters()).device
        _dataloader = (
            self.calibration_dataloader_
            if self.num_calibration_steps is None
            else cycle(self.calibration_dataloader_)
        )

        for batch_idx, batch in enumerate(_dataloader):
            if self.num_calibration_steps and batch_idx >= self.num_calibration_steps:
                break
            batch = tensors_to_device(batch, model_device)
            with torch.no_grad():
                forward_fn(batch, module=module)

        if module_training:
            module.train()
        else:
            module.apply(torch.quantization.disable_observer)
