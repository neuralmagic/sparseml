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
Sensitivity analysis implementations for increasing activation sparsity by using FATReLU
"""

from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle

from sparseml.pytorch.nn.fatrelu import FATReLU, convert_relus_to_fat
from sparseml.pytorch.optim.analyzer_as import ModuleASAnalyzer
from sparseml.pytorch.utils import (
    LossWrapper,
    ModuleRunResults,
    ModuleTester,
    get_layer,
    model_to_device,
)


__all__ = ["ASLayerTracker", "LayerBoostResults", "ModuleASOneShootBooster"]


class ASLayerTracker(object):
    """
    An implementation for tracking activation sparsity properties for a module.

    :param layer: the module to track activation sparsity for
    :param track_input: track the input sparsity for the module
    :param track_output: track the output sparsity for the module
    :param input_func: the function to call on input to the layer
        and receives the input tensor
    :param output_func: the function to call on output to the layer
        and receives the output tensor
    """

    def __init__(
        self,
        layer: Module,
        track_input: bool = False,
        track_output: bool = False,
        input_func: Union[None, Callable] = None,
        output_func: Union[None, Callable] = None,
    ):
        super().__init__()
        self._layer = layer
        self._track_input = track_input
        self._track_output = track_output
        self._input_func = input_func
        self._output_func = output_func

        self._enabled = False
        self._tracked_input = {}
        self._tracked_output = {}
        self._hook_handle = None  # type: RemovableHandle

    def __del__(self):
        self._disable_hooks()

    def enable(self):
        """
        Enable the forward hooks to the layer
        """
        if not self._enabled:
            self._enabled = True
            self._enable_hooks()

        self.clear()

    def disable(self):
        """
        Disable the forward hooks for the layer
        """
        if self._enabled:
            self._enabled = False
            self._disable_hooks()

        self.clear()

    def clear(self):
        """
        Clear out current results for the model
        """
        self._tracked_input.clear()
        self._tracked_output.clear()

    @property
    def tracked_input(self):
        """
        :return: the current tracked input results
        """
        return self._tracked_input

    @property
    def tracked_output(self):
        """
        :return: the current tracked output results
        """
        return self._tracked_output

    def _enable_hooks(self):
        if self._hook_handle is not None:
            return

        def _forward_hook(
            _mod: Module,
            _inp: Union[Tensor, Tuple[Tensor]],
            _out: Union[Tensor, Tuple[Tensor]],
        ):
            if self._track_input:
                tracked = _inp

                if self._input_func is not None:
                    tracked = self._input_func(_inp)

                key = (
                    "cpu"
                    if not tracked.is_cuda
                    else "cuda:{}".format(tracked.get_device())
                )
                self._tracked_input[key] = tracked

            if self._track_output:
                tracked = _out

                if self._output_func is not None:
                    tracked = self._output_func(_out)

                key = (
                    "cpu"
                    if not tracked.is_cuda
                    else "cuda:{}".format(tracked.get_device())
                )
                self._tracked_output[key] = tracked

        self._hook_handle = self._layer.register_forward_hook(_forward_hook)

    def _disable_hooks(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class LayerBoostResults(object):
    """
    Results for a specific threshold set in a FATReLU layer.

    :param name: the name of the layer the results are for
    :param threshold: the threshold used in the FATReLU layer
    :param boosted_as: the measured activation sparsity after threshold is applied
    :param boosted_loss: the measured loss after threshold is applied
    :param baseline_as: the measured activation sparsity before threshold is applied
    :param baseline_loss: the measured loss before threshold is applied
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        boosted_as: Tensor,
        boosted_loss: ModuleRunResults,
        baseline_as: Tensor,
        baseline_loss: ModuleRunResults,
    ):
        self._name = name
        self._threshold = threshold
        self._boosted_as = boosted_as
        self._boosted_loss = boosted_loss
        self._baseline_as = baseline_as
        self._baseline_loss = baseline_loss

    @property
    def name(self) -> str:
        """
        :return: the name of the layer the results are for
        """
        return self._name

    @property
    def threshold(self) -> float:
        """
        :return: the threshold used in the FATReLU layer
        """
        return self._threshold

    @property
    def boosted_as(self) -> Tensor:
        """
        :return: the measured activation sparsity after threshold is applied
        """
        return self._boosted_as

    @property
    def boosted_loss(self) -> ModuleRunResults:
        """
        :return: the measured loss after threshold is applied
        """
        return self._boosted_loss

    @property
    def baseline_as(self) -> Tensor:
        """
        :return: the measured activation sparsity before threshold is applied
        """
        return self._baseline_as

    @property
    def baseline_loss(self) -> ModuleRunResults:
        """
        :return: the measured loss before threshold is applied
        """
        return self._baseline_loss


class ModuleASOneShootBooster(object):
    """
    Implementation class for boosting the activation sparsity in a given module
    using FATReLUs.
    Programmatically goes through and figures out the best thresholds to limit loss
    based on provided parameters.

    :param module: the module to boost
    :param device: the device to run the analysis on; ex [cpu, cuda, cuda:1]
    :param dataset: the dataset used to evaluate the boosting on
    :param batch_size: the batch size to run through the module in test mode
    :param loss: the loss function to use for calculations
    :param data_loader_kwargs: any keyword arguments to supply to a the
        DataLoader constructor
    """

    def __init__(
        self,
        module: Module,
        device: str,
        dataset: Dataset,
        batch_size: int,
        loss: LossWrapper,
        data_loader_kwargs: Dict,
    ):
        self._module = module
        self._device = device
        self._dataset = dataset
        self._batch_size = batch_size
        self._loss = loss
        self._dataloader_kwargs = data_loader_kwargs if data_loader_kwargs else {}

    def run_layers(
        self,
        layers: List[str],
        max_target_metric_loss: float,
        metric_key: str,
        metric_increases: bool,
        precision: float = 0.001,
    ) -> Dict[str, LayerBoostResults]:
        """
        Run the booster for the specified layers.

        :param layers: names of the layers to run boosting on
        :param max_target_metric_loss: the max loss in the target metric that
            can happen while boosting
        :param metric_key: the name of the metric to evaluate while boosting;
            ex: [__loss__, top1acc, top5acc]. Must exist in the LossWrapper
        :param metric_increases: True if the metric increases for worse loss such as in
            a CrossEntropyLoss, False if the metric decreases for worse such as in
            accuracy
        :param precision: the precision to check the results to. Larger values here will
            give less precise results but won't take as long
        :return: The results for the boosting
        """
        fat_relus = convert_relus_to_fat(
            self._module, inplace=True
        )  # type: Dict[str, FATReLU]

        for layer in layers:
            if layer not in fat_relus:
                raise KeyError(
                    (
                        "layer {} was specified in the config but is not a "
                        "boostable layer in the module (ie not a relu)"
                    ).format(layer)
                )

        module, device, device_ids = model_to_device(self._module, self._device)
        results = {}
        min_thresh = 0.0
        max_thresh = 1.0

        baseline_res, _ = self._measure_layer(None, module, device, "baseline loss run")

        for layer in layers:
            results[layer] = self._binary_search_fat(
                layer,
                module,
                device,
                min_thresh,
                max_thresh,
                max_target_metric_loss,
                metric_key,
                metric_increases,
                precision,
            )

        boosted_res, _ = self._measure_layer(None, module, device, "boosted loss run")
        results["__module__"] = LayerBoostResults(
            "__module__",
            -1.0,
            torch.tensor(-1.0),
            boosted_res,
            torch.tensor(-1.0),
            baseline_res,
        )

        return results

    def _binary_search_fat(
        self,
        layer: str,
        module: Module,
        device: str,
        min_thresh: float,
        max_thresh: float,
        max_target_metric_loss: float,
        metric_key: str,
        metric_increases: bool,
        precision: float = 0.001,
    ):
        print(
            "\n\n\nstarting binary search for layer {} between ({}, {})...".format(
                layer, min_thresh, max_thresh
            )
        )
        base_res, base_as = self._measure_layer(
            layer, module, device, "baseline for layer: {}".format(layer)
        )

        fat_relu = get_layer(layer, module)  # type: FATReLU
        init_thresh = fat_relu.get_threshold()

        if min_thresh > init_thresh:
            min_thresh = init_thresh

        while True:
            thresh = ModuleASOneShootBooster._get_mid_point(min_thresh, max_thresh)
            fat_relu.set_threshold(thresh)

            thresh_res, thresh_as = self._measure_layer(
                layer,
                module,
                device,
                "threshold for layer: {} @ {:.4f} ({:.4f}, {:.4f})".format(
                    layer, thresh, min_thresh, max_thresh
                ),
            )

            if ModuleASOneShootBooster._passes_loss(
                base_res,
                thresh_res,
                max_target_metric_loss,
                metric_key,
                metric_increases,
            ):
                min_thresh = thresh
                print(
                    "loss check passed for max change: {:.4f}".format(
                        max_target_metric_loss
                    )
                )
                print(
                    "   current loss: {:.4f} baseline loss: {:.4f}".format(
                        thresh_res.result_mean(metric_key),
                        base_res.result_mean(metric_key),
                    )
                )
                print(
                    "   current AS: {:.4f} baseline AS: {:.4f}".format(
                        thresh_as, base_as
                    )
                )
            else:
                max_thresh = thresh
                print(
                    "loss check failed for max change: {:.4f}".format(
                        max_target_metric_loss
                    )
                )
                print(
                    "   current loss: {:.4f} baseline loss: {:.4f}".format(
                        thresh_res.result_mean(metric_key),
                        base_res.result_mean(metric_key),
                    )
                )
                print(
                    "   current AS: {:.4f} baseline AS: {:.4f}".format(
                        thresh_as, base_as
                    )
                )

            if max_thresh - min_thresh <= precision:
                break

        print("completed binary search for layer {}")
        print("   found threshold: {}".format(thresh))
        print(
            "   AS delta: {:.4f} ({:.4f} => {:.4f})".format(
                thresh_as - base_as, base_as, thresh_as
            )
        )

        return LayerBoostResults(
            layer, thresh, thresh_as, thresh_res, base_as, base_res
        )

    def _measure_layer(
        self, layer: Union[str, None], module: Module, device: str, desc: str
    ) -> Tuple[ModuleRunResults, Tensor]:
        layer = get_layer(layer, module) if layer else None
        as_analyzer = None

        if layer:
            as_analyzer = ModuleASAnalyzer(layer, dim=None, track_outputs_sparsity=True)
            as_analyzer.enable()

        tester = ModuleTester(module, device, self._loss)
        data_loader = DataLoader(
            self._dataset, self._batch_size, **self._dataloader_kwargs
        )
        results = tester.run(
            data_loader, desc=desc, show_progress=True, track_results=True
        )

        if as_analyzer:
            as_analyzer.disable()

        return results, as_analyzer.outputs_sparsity_mean if as_analyzer else None

    @staticmethod
    def _get_mid_point(start: float, end: float) -> float:
        return (end - start) / 2.0 + start

    @staticmethod
    def _passes_loss(
        base_res: ModuleRunResults,
        thresh_res: ModuleRunResults,
        max_target_metric_loss: float,
        metric_key: str,
        metric_increases: bool,
    ) -> bool:
        diff = (
            thresh_res.result_mean(metric_key) - base_res.result_mean(metric_key)
            if metric_increases
            else base_res.result_mean(metric_key) - thresh_res.result_mean(metric_key)
        )

        return diff < max_target_metric_loss
