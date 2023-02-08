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
Code related to analyzing activation sparsity within PyTorch neural networks.
More information can be found in the paper
`here <https://arxiv.org/abs/1705.01626>`__.
"""

from enum import Enum
from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from sparseml.pytorch.utils import get_layer, tensor_sample, tensor_sparsity


__all__ = ["ASResultType", "ModuleASAnalyzer"]


class ASResultType(Enum):
    """
    Result type to track for activation sparsity.
    """

    inputs_sparsity = "inputs_sparsity"
    inputs_sample = "inputs_sample"
    outputs_sparsity = "outputs_sparsity"
    outputs_sample = "outputs_sample"


class ModuleASAnalyzer(object):
    """
    An analyzer implementation used to monitor the activation sparsity with a module.
    Generally used to monitor an individual layer.

    :param module: The module to analyze activation sparsity for
    :param dim: Any dims within the tensor such as across batch,
        channel, etc. Ex: 0 for batch, 1 for channel, [0, 1] for batch and channel
    :param track_inputs_sparsity: True to track the input sparsity to the module,
        False otherwise
    :param track_outputs_sparsity: True to track the output sparsity to the module,
        False otherwise
    :param inputs_sample_size: The number of samples to grab from the input tensor
        on each forward pass. If <= 0, then will not sample any values.
    :param outputs_sample_size: The number of samples to grab from the output tensor
        on each forward pass. If <= 0, then will not sample any values.
    :param enabled: True to enable the hooks for analyzing and actively track,
        False to disable and not track
    """

    @staticmethod
    def analyze_layers(
        module: Module,
        layers: List[str],
        dim: Union[None, int, Tuple[int, ...]] = None,
        track_inputs_sparsity: bool = False,
        track_outputs_sparsity: bool = False,
        inputs_sample_size: int = 0,
        outputs_sample_size: int = 0,
        enabled: bool = True,
    ):
        """
        :param module: the module to analyze multiple layers activation sparsity in
        :param layers: the names of the layers to analyze (from module.named_modules())
        :param dim: Any dims within the tensor such as across batch,
            channel, etc. Ex: 0 for batch, 1 for channel, [0, 1] for batch and channel
        :param track_inputs_sparsity: True to track the input sparsity to the module,
            False otherwise
        :param track_outputs_sparsity: True to track the output sparsity to the module,
            False otherwise
        :param inputs_sample_size: The number of samples to grab from the input tensor
            on each forward pass. If <= 0, then will not sample any values.
        :param outputs_sample_size: The number of samples to grab from the output tensor
            on each forward pass. If <= 0, then will not sample any values.
        :param enabled: True to enable the hooks for analyzing and actively track,
            False to disable and not track
        :return: a list of the created analyzers, matches the ordering in layers
        """
        analyzed = []

        for layer_name in layers:
            layer = get_layer(layer_name, module)
            analyzed.append(
                ModuleASAnalyzer(
                    layer,
                    dim,
                    track_inputs_sparsity,
                    track_outputs_sparsity,
                    inputs_sample_size,
                    outputs_sample_size,
                    enabled,
                )
            )

        return analyzed

    def __init__(
        self,
        module: Module,
        dim: Union[None, int, Tuple[int, ...]] = None,
        track_inputs_sparsity: bool = False,
        track_outputs_sparsity: bool = False,
        inputs_sample_size: int = 0,
        outputs_sample_size: int = 0,
        enabled: bool = True,
    ):
        self._module = module
        self._dim = dim
        self._track_inputs_sparsity = track_inputs_sparsity
        self._track_outputs_sparsity = track_outputs_sparsity
        self._inputs_sample_size = inputs_sample_size
        self._outputs_sample_size = outputs_sample_size
        self._enabled = False

        self._inputs_sparsity = []  # type: List[Tensor]
        self._inputs_sample = []  # type: List[Tensor]
        self._outputs_sparsity = []  # type: List[Tensor]
        self._outputs_sample = []  # type: List[Tensor]
        self._pre_hook_handle = None  # type: RemovableHandle
        self._hook_handle = None  # type: RemovableHandle

        if enabled:
            self.enable()

    def __del__(self):
        self._disable_hooks()

    def __str__(self):
        return (
            "module: {}, dim: {}, track_inputs_sparsity: {},"
            " track_outputs_sparsity: {}, inputs_sample_size: {},"
            " outputs_sample_size: {}, enabled: {}"
        ).format(
            self._module,
            self._dim,
            self._track_inputs_sparsity,
            self._track_outputs_sparsity,
            self._inputs_sample_size,
            self._outputs_sample_size,
            self._enabled,
        )

    @property
    def module(self) -> Module:
        return self._module

    @property
    def dim(self) -> Union[None, int, Tuple[int, ...]]:
        return self._dim

    @property
    def track_inputs_sparsity(self) -> bool:
        return self._track_inputs_sparsity

    @track_inputs_sparsity.setter
    def track_inputs_sparsity(self, value: bool):
        self._track_inputs_sparsity = value

    @property
    def track_outputs_sparsity(self) -> bool:
        return self._track_outputs_sparsity

    @track_outputs_sparsity.setter
    def track_outputs_sparsity(self, value: bool):
        self._track_outputs_sparsity = value

    @property
    def inputs_sample_size(self) -> int:
        return self._inputs_sample_size

    @inputs_sample_size.setter
    def inputs_sample_size(self, value: int):
        self._inputs_sample_size = value

    @property
    def outputs_sample_size(self) -> int:
        return self._outputs_sample_size

    @outputs_sample_size.setter
    def outputs_sample_size(self, value: int):
        self._outputs_sample_size = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def inputs_sparsity(self) -> List[Tensor]:
        return self.results(ASResultType.inputs_sparsity)

    @property
    def inputs_sparsity_mean(self) -> Tensor:
        return self.results_mean(ASResultType.inputs_sparsity)

    @property
    def inputs_sparsity_std(self) -> Tensor:
        return self.results_std(ASResultType.inputs_sparsity)

    @property
    def inputs_sparsity_max(self) -> Tensor:
        return self.results_max(ASResultType.inputs_sparsity)

    @property
    def inputs_sparsity_min(self) -> Tensor:
        return self.results_min(ASResultType.inputs_sparsity)

    @property
    def inputs_sample(self) -> List[Tensor]:
        return self.results(ASResultType.inputs_sample)

    @property
    def inputs_sample_mean(self) -> Tensor:
        return self.results_mean(ASResultType.inputs_sample)

    @property
    def inputs_sample_std(self) -> Tensor:
        return self.results_std(ASResultType.inputs_sample)

    @property
    def inputs_sample_max(self) -> Tensor:
        return self.results_max(ASResultType.inputs_sample)

    @property
    def inputs_sample_min(self) -> Tensor:
        return self.results_min(ASResultType.inputs_sample)

    @property
    def outputs_sparsity(self) -> List[Tensor]:
        return self.results(ASResultType.outputs_sparsity)

    @property
    def outputs_sparsity_mean(self) -> Tensor:
        return self.results_mean(ASResultType.outputs_sparsity)

    @property
    def outputs_sparsity_std(self) -> Tensor:
        return self.results_std(ASResultType.outputs_sparsity)

    @property
    def outputs_sparsity_max(self) -> Tensor:
        return self.results_max(ASResultType.outputs_sparsity)

    @property
    def outputs_sparsity_min(self) -> Tensor:
        return self.results_min(ASResultType.outputs_sparsity)

    @property
    def outputs_sample(self) -> List[Tensor]:
        return self.results(ASResultType.outputs_sample)

    @property
    def outputs_sample_mean(self) -> Tensor:
        return self.results_mean(ASResultType.outputs_sample)

    @property
    def outputs_sample_std(self) -> Tensor:
        return self.results_std(ASResultType.outputs_sample)

    @property
    def outputs_sample_max(self) -> Tensor:
        return self.results_max(ASResultType.outputs_sample)

    @property
    def outputs_sample_min(self) -> Tensor:
        return self.results_min(ASResultType.outputs_sample)

    def clear(self, specific_result_type: Union[None, ASResultType] = None):
        if (
            specific_result_type is None
            or specific_result_type == ASResultType.inputs_sparsity
        ):
            self._inputs_sparsity.clear()

        if (
            specific_result_type is None
            or specific_result_type == ASResultType.inputs_sample
        ):
            self._inputs_sample.clear()

        if (
            specific_result_type is None
            or specific_result_type == ASResultType.outputs_sparsity
        ):
            self._outputs_sparsity.clear()

        if (
            specific_result_type is None
            or specific_result_type == ASResultType.outputs_sample
        ):
            self._outputs_sample.clear()

    def enable(self):
        if not self._hook_handle or not self._pre_hook_handle:
            self._enabled = True
            self._enable_hooks()

    def disable(self):
        if self._hook_handle or self._pre_hook_handle:
            self._enabled = False
            self._disable_hooks()

    def results(self, result_type: ASResultType) -> List[Tensor]:
        if result_type == ASResultType.inputs_sparsity:
            res = self._inputs_sparsity
        elif result_type == ASResultType.inputs_sample:
            res = self._inputs_sample
        elif result_type == ASResultType.outputs_sparsity:
            res = self._outputs_sparsity
        elif result_type == ASResultType.outputs_sample:
            res = self._outputs_sample
        else:
            raise ValueError("result_type of {} is not supported".format(result_type))

        if not res:
            res = torch.tensor([])

        res = [r if r.shape else r.unsqueeze(0) for r in res]

        return res

    def results_mean(self, result_type: ASResultType) -> Tensor:
        results = self.results(result_type)

        return torch.mean(torch.cat(results), dim=0)

    def results_std(self, result_type: ASResultType) -> Tensor:
        results = self.results(result_type)

        return torch.std(torch.cat(results), dim=0)

    def results_max(self, result_type: ASResultType) -> Tensor:
        results = self.results(result_type)

        return torch.max(torch.cat(results))

    def results_min(self, result_type: ASResultType) -> Tensor:
        results = self.results(result_type)

        return torch.min(torch.cat(results))

    def _enable_hooks(self):
        def _forward_pre_hook(_mod: Module, _inp: Union[Tensor, Tuple[Tensor]]):
            if not isinstance(_inp, Tensor):
                _inp = _inp[0]

            if self.track_inputs_sparsity:
                result = tensor_sparsity(_inp, dim=self.dim)
                sparsities = result.detach_().cpu()
                self._inputs_sparsity.append(sparsities)

            if self.inputs_sample_size > 0:
                result = tensor_sample(_inp, self.inputs_sample_size, dim=self.dim)
                samples = result.detach_().cpu()
                self._inputs_sample.append(samples)

        def _forward_hook(
            _mod: Module,
            _inp: Union[Tensor, Tuple[Tensor]],
            _out: Union[Tensor, Tuple[Tensor]],
        ):
            if not isinstance(_out, Tensor):
                _out = _out[0]

            if self.track_outputs_sparsity:
                result = tensor_sparsity(_out, dim=self.dim)
                sparsities = result.detach_().cpu()
                self._outputs_sparsity.append(sparsities)

            if self.outputs_sample_size > 0:
                result = tensor_sample(_out, self.outputs_sample_size, dim=self.dim)
                samples = result.detach_().cpu()
                self._outputs_sample.append(samples)

        self._pre_hook_handle = self.module.register_forward_pre_hook(_forward_pre_hook)
        self._hook_handle = self.module.register_forward_hook(_forward_hook)

    def _disable_hooks(self):
        if self._pre_hook_handle is not None:
            self._pre_hook_handle.remove()
            self._pre_hook_handle = None

        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
