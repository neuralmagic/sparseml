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
from typing import Any, Optional

from torch.nn import Module

from compressed_tensors.quantization import (
    apply_quantization_config,
    freeze_module_quantization,
    set_module_for_calibration,
)
from compressed_tensors.quantization.observers.helpers import get_observer_token_count
from sparseml.core import Event, EventType, State
from sparseml.modifiers.quantization_vllm.base import vLLMQuantizationModifier
from sparseml.modifiers.utils.pytorch_helpers import run_calibration_forward


_LOGGER = logging.getLogger(__name__)


class vLLMQuantizationModifierPyTorch(vLLMQuantizationModifier):
    """
    PyTorch specific implementation of vLLMQuantizationModifier

    Enables post training quantization (PTQ) and quantization aware training (QAT) for a
    given module or its submodules. After calibration (PTQ) or the start epoch (QAT),
    the specified module(s) forward pass will emulate quantized execution and the
    modifier will be enabled until training is completed.

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    """

    calibration_dataloader_: Any = None
    calibration_function_: Any = None

    def on_initialize_structure(self, state: State, **kwargs):
        module = state.model.model
        self._apply_modifier_to_model(module)
        module.apply(freeze_module_quantization)

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.end and self.end != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1 or None. Given {}".format(self.end)
            )

        self.calibration_dataloader_ = state.data.calib
        module = state.model.model

        # intialize quantization in appropriate modules
        self._apply_modifier_to_model(module)

        if self.calculate_start() == -1:  # one-shot
            module.apply(set_module_for_calibration)
            self._calibrate_if_possible(module)
            self._check_token_distribution(
                module, threshold=kwargs.get("min_tokens_per_group", None)
            )
            module.apply(freeze_module_quantization)

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        module = state.model.model
        module.apply(set_module_for_calibration)

    def on_update(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.BATCH_START:
            if self.check_should_disable_observer(event):
                module = state.model.model
                module.apply(freeze_module_quantization)

    def on_end(self, state: State, event: Event, **kwargs):
        module = state.model.model
        module.apply(freeze_module_quantization)

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def _apply_modifier_to_model(self, model: Module):
        modifier_as_config = self.create_init_config()
        apply_quantization_config(model, modifier_as_config)

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

    def _check_token_distribution(
        self, model: Module, threshold: Optional[float] = None
    ):
        """
        A helper function that warns when a module has seen
        fewer than threshold % of all the tokens in the batch.

        Checks are only triggered if threshold is not None.

        :param batch: the batch of tokens (batch_size, sequence_length)
        :param model: the model to investigate
        :param threshold: the minimum percentage of tokens
            (out of all the tokens in a batch) a module should
            receive during each forward pass of the calibration
        """
        if threshold is None:
            _LOGGER.debug("Skipping token distribution check. Threshold is None.")
            return

        all_tokens = self.calibration_dataloader_.dataset["input_ids"]
        total_token_count = sum(len(sample) for sample in all_tokens)
        counter = get_observer_token_count(model)
        for module_name, token_count in counter.items():
            if token_count is None:
                # the module has not been observed
                # or its token_count is not being recorded
                # by the observer (refer to the observer's
                # implementation in the source code)
                continue
            if token_count / total_token_count < threshold:
                _LOGGER.warning(
                    f"The module_name: {module_name} "
                    f"received less than {int(threshold * 100)}% "
                    "of calibration batch tokens "
                    f"({token_count}/{total_token_count} tokens). "
                    "This could result may harm the quantization quality."
                )
