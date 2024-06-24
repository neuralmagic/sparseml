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

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from sparseml.core import Event, Modifier


__all__ = ["LegacyQuantizationModifier"]


class LegacyQuantizationModifier(Modifier):
    """
    Enables quantization aware training (QAT) for a given module or its submodules
    After the start epoch, the specified module(s) forward pass will emulate
    quantized execution and the modifier will be enabled until training is completed.

    | Sample yaml:
    |   LegacyQuantizationModifier:
    |       start: 0.0
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
    """

    ignore: Optional[List[str]] = None
    disable_quantization_observer_epoch: Optional[float] = None
    freeze_bn_stats_epoch: Optional[float] = None
    model_fuse_fn_name: Optional[str] = None
    model_fuse_fn_kwargs: Optional[Dict[str, Any]] = None
    num_calibration_steps: Optional[int] = None
    post_oneshot_calibration: Optional[bool] = False
    strict: bool = True

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_fuse_fn_kwargs is None:
            self.model_fuse_fn_kwargs = {}
        if self.ignore is None:
            self.ignore = []

    def calculate_freeze_bn_stats_epoch(self) -> float:
        """
        Get the epoch at which we want to stop updating batch normalization stats

        :return: freeze_bn_stats_epoch if set, else -1
        """
        return (
            self.freeze_bn_stats_epoch if self.freeze_bn_stats_epoch is not None else -1
        )

    def check_should_freeze_bn_stats(self, event: Event) -> bool:
        """
        Given the current index, determine if we should freeze batch normalization stats

        :param event: Event to get index from
        :return: True if stats should be frozen, False otherwise
        """
        freeze_epoch = self.calculate_freeze_bn_stats_epoch()
        if freeze_epoch == -1:
            return False
        if event.current_index >= freeze_epoch:
            return True
        return False

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
