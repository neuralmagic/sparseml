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

from typing import Dict, List, Optional

from pydantic import Field

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from sparseml.core import Event, Modifier


__all__ = ["QuantizationModifier"]


class QuantizationModifier(Modifier):
    """
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

    config_groups: Dict[str, QuantizationScheme]
    ignore: List[str] = Field(default_factory=list)
    disable_quantization_observer_epoch: Optional[float] = None
    num_calibration_steps: Optional[int] = None

    def create_init_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            config_groups=self.config_groups,
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=self.ignore,
        )

    def calculate_disable_observer_epoch(self) -> float:
        """
        Get the epoch at which we want to disable to quantization observer
        :return epoch to disable at, or -1 if it is not set
        """
        return (
            self.disable_quantization_observer_epoch
            if self.disable_quantization_observer_epoch is not None
            else -1
        )

    def check_should_disable_observer(self, event: Event) -> bool:
        """
        Given the current index, determine if we should disable the observer

        :param event: Event to get index from
        :return: True if observer should be disabled, False otherwise
        """
        disable_epoch = self.calculate_disable_observer_epoch()
        if disable_epoch == -1:
            return False
        if event.current_index >= disable_epoch:
            return True
        return False
