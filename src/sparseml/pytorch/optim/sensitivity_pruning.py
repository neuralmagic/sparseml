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
Sensitivity analysis implementations for kernel sparsity on Modules against loss funcs.
"""

from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from sparseml.optim import (
    PruningLossSensitivityAnalysis,
    default_pruning_sparsities_loss,
)
from sparseml.pytorch.optim.mask_creator_pruning import UnstructuredPruningMaskCreator
from sparseml.pytorch.optim.mask_pruning import ModuleParamPruningMask
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    BaseLogger,
    LossWrapper,
    ModuleRunFuncs,
    ModuleTester,
    get_prunable_layers,
    infinite_data_loader,
)


__all__ = [
    "model_prunability_magnitude",
    "pruning_loss_sens_magnitude",
    "pruning_loss_sens_one_shot",
]


def model_prunability_magnitude(module: Module):
    """
    Calculate the approximate sensitivity for an overall model.
    Range of the values are not scaled to anything, so must be taken in context
    with other known models.

    :param module: the model to calculate the sensitivity for
    :return: the approximated sensitivity
    """
    prunable = get_prunable_layers(module)
    tensors = []

    for (name, layer) in prunable:
        weight = getattr(layer, "weight")
        values = weight.view(-1).abs()
        tensors.append(values)

    all_weights = torch.cat(tensors)
    avg = all_weights.mean().item()

    return avg


def pruning_loss_sens_magnitude(
    module: Module,
    sparsity_levels: Union[
        List[float], Tuple[float, ...]
    ] = default_pruning_sparsities_loss(True),
) -> PruningLossSensitivityAnalysis:
    """
    Approximated kernel sparsity (pruning) loss analysis for a given model.
    Returns the results for each prunable param (conv, linear) in the model.

    :param module: the model to calculate the sparse sensitivity analysis for
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :return: the analysis results for the model
    """
    prunable = get_prunable_layers(module)
    analysis = PruningLossSensitivityAnalysis()

    for index, (name, layer) in enumerate(prunable):
        weight = getattr(layer, "weight")
        name = "{}.weight".format(name)
        values, _ = weight.view(-1).abs().sort()
        prev_index = 0

        for sparsity in sparsity_levels:
            val_index = round(sparsity * len(values))

            if val_index >= len(values):
                val_index = len(values) - 1

            if sparsity <= 1e-9:
                baseline = True
                sparsity = 0.0
                sparse_avg = 0.0
            else:
                baseline = False

                if val_index > prev_index:
                    sparse_avg = values[prev_index:val_index].mean().item()
                    prev_index = val_index
                else:
                    sparse_avg = values[val_index].item()
                    prev_index = val_index + 1

            analysis.add_result(None, name, index, sparsity, sparse_avg, baseline)

    return analysis


def _sensitivity_callback(
    prunable_layers: List[Tuple[str, Module]],
    sparsity_levels: List[int],
    steps_per_measurement: int,
    analysis: PruningLossSensitivityAnalysis,
    loss_key: str,
) -> Callable:
    measurement_steps = 0
    layer_index = -1
    sparsity_index = -1
    current_mask = None

    def complete_measurement():
        """
        Uses complete_measurement to handle when all of the required steps have been
        taken for a given layer and sparsity level.
        This handles incrementing to the next sparsity level.
        If all sparsity levels are complete,
        increments to the next layer and starts from the initial sparsity level.

        Should only be invoked when all measurements have been taken.
        """

        nonlocal measurement_steps
        nonlocal layer_index
        nonlocal sparsity_index
        nonlocal current_mask

        measurement_steps = 0
        sparsity_index += 1

        if 0 <= sparsity_index < len(sparsity_levels) and 0 <= layer_index < len(
            prunable_layers
        ):
            # increment sparsity level for current layer
            current_mask.set_param_masks_from_sparsity(sparsity_levels[sparsity_index])
        else:
            # go to next layer
            sparsity_index = 0
            layer_index += 1

            if current_mask:
                current_mask.enabled = False
                current_mask.reset()
                del current_mask
                current_mask = None

            if layer_index < len(prunable_layers):
                current_mask = ModuleParamPruningMask(
                    [prunable_layers[layer_index][1]],
                    store_init=True,
                    mask_creator=UnstructuredPruningMaskCreator(),
                )
                current_mask.enabled = True

                if sparsity_levels[sparsity_index] > 0.0:
                    current_mask.set_param_masks_from_sparsity(
                        sparsity_levels[sparsity_index]
                    )

    complete_measurement()

    def batch_end(
        epoch: int,
        step: int,
        batch_size: int,
        data: Any,
        pred: Any,
        losses: Any,
    ):
        nonlocal measurement_steps
        measurement_steps += 1

        if layer_index < len(prunable_layers):
            analysis.add_result(
                None,
                "{}.weight".format(prunable_layers[layer_index][0]),
                sparsity_index,
                sparsity_levels[sparsity_index],
                losses[loss_key].item(),
                baseline=sparsity_levels[sparsity_index] < 1e-9,
            )

        if measurement_steps >= steps_per_measurement:
            complete_measurement()

    return batch_end


def pruning_loss_sens_one_shot(
    module: Module,
    data: DataLoader,
    loss: Union[LossWrapper, Callable[[Any, Any], Tensor]],
    device: str,
    steps_per_measurement: int,
    sparsity_levels: List[int] = default_pruning_sparsities_loss(False),
    loss_key: str = DEFAULT_LOSS_KEY,
    tester_run_funcs: ModuleRunFuncs = None,
    tester_loggers: List[BaseLogger] = None,
    show_progress: bool = True,
) -> PruningLossSensitivityAnalysis:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    It does not retrain, and instead puts the model to eval mode.
    Moves layer by layer to calculate the sensitivity analysis for each and
    resets the previously run layers.
    Note, by default it caches the data.
    This means it is not parallel for data loading and the first run can take longer.
    Subsequent sparsity checks for layers and levels will be much faster.

    :param module: the module to run the kernel sparsity sensitivity analysis over
        will extract all prunable layers out
    :param data: the data to run through the module for calculating the sensitivity
        analysis
    :param loss: the loss function to use for the sensitivity analysis
    :param device: the device to run the analysis on; ex: cpu, cuda
    :param steps_per_measurement: the number of samples or items to take for each
        measurement at each sparsity lev
    :param sparsity_levels: the sparsity levels to check for each layer to calculate
        sensitivity
    :param loss_key: the key for the loss function to track in the returned dict
    :param tester_run_funcs: override functions to use in the ModuleTester that runs
    :param tester_loggers: loggers to log data to while running the analysis
    :param show_progress: track progress of the runs if True
    :return: the sensitivity results for every layer that is prunable
    """
    analysis = PruningLossSensitivityAnalysis()
    tester = ModuleTester(
        module,
        device,
        loss,
        loggers=tester_loggers,
        log_summary=False,
        log_steps=max(1, round(steps_per_measurement / 10)),
    )
    layers = get_prunable_layers(module)
    batch_end = _sensitivity_callback(
        layers, sparsity_levels, steps_per_measurement, analysis, loss_key
    )
    batch_end_hook = tester.run_hooks.register_batch_end_hook(batch_end)
    if tester_run_funcs is not None:
        tester.run_funcs.copy(tester_run_funcs)

    data_loader = infinite_data_loader(
        data, early_stop_steps=steps_per_measurement, cache=True
    )
    tester.run(
        data_loader,
        desc="KS Analysis",
        show_progress=show_progress,
        track_results=False,
        max_steps=steps_per_measurement * len(sparsity_levels) * len(layers),
    )
    batch_end_hook.remove()

    return analysis
