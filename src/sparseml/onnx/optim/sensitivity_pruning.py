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
Sensitivity analysis implementations for kernel sparsity on Models against loss funcs.
"""

import logging
import numbers
from typing import Any, Generator, List, NamedTuple, Tuple, Union

import numpy
from onnx import ModelProto
from tqdm import auto

from sparseml.onnx.utils import (
    check_load_model,
    extract_node_id,
    get_node_params,
    get_prunable_nodes,
)
from sparseml.optim import (
    PruningLossSensitivityAnalysis,
    PruningPerfSensitivityAnalysis,
    PruningSensitivityResult,
    default_pruning_sparsities_loss,
)
from sparseml.utils import flatten_iterable


_LOGGER = logging.getLogger(__name__)


__all__ = [
    "pruning_loss_sens_approx",
    "pruning_loss_sens_magnitude",
    "pruning_loss_sens_magnitude_iter",
    "PruningLossSensitivityAnalysis",
    "PruningPerfSensitivityAnalysis",
    "PruningSensitivityResult",
    "KSSensitivityProgress",
]


"""
Simple named tuple for returning the progress of KS sensitivity analysis
"""
KSSensitivityProgress = NamedTuple(
    "KSSensitivityProgress",
    [("current", int), ("current_metadata", Any), ("total", int), ("val", float)],
)


def pruning_loss_sens_approx(
    input_shape: Union[None, List[int], List[List[int]]],
    output_shape: Union[None, List[int]],
    params: int,
    apply_shape_change_mult: bool = True,
) -> float:
    """
    Approximate the pruning sensitivity of a Neural Network's layer
    based on the params and metadata for a given layer

    :param input_shape: the input shape to the layer
    :param output_shape: the output shape from the layer
    :param params: the number of params in the layer
    :param apply_shape_change_mult: True to adjust the sensitivity based on
        a weight derived from a change in input to output shape
        (any change is considered to be more sensitive), False to not apply
    :return: the approximated pruning sensitivity for the layer's settings
    """
    if not params:
        return 0.0

    if input_shape:
        input_shape = flatten_iterable(input_shape)
        input_shape = [
            size for size in input_shape if size and isinstance(size, numbers.Number)
        ]

    input_volume = 0 if not input_shape else numpy.prod(input_shape).item()

    if output_shape:
        output_shape = flatten_iterable(output_shape)
        output_shape = [
            size for size in output_shape if size and isinstance(size, numbers.Number)
        ]

    output_volume = 0 if not output_shape else numpy.prod(output_shape).item()
    total_volume = input_volume + output_volume

    features_per_params = total_volume / float(params)
    shape_change_mult = (
        1.0
        if not apply_shape_change_mult or not input_volume or not output_volume
        else max(input_volume / output_volume, output_volume / input_volume)
    )

    return features_per_params * shape_change_mult


def pruning_loss_sens_magnitude_iter(
    model: Union[str, ModelProto],
    sparsity_levels: Union[
        List[float], Tuple[float, ...]
    ] = default_pruning_sparsities_loss(True),
) -> Generator[
    Tuple[PruningLossSensitivityAnalysis, KSSensitivityProgress], None, None
]:
    """
    Approximated kernel sparsity (pruning) loss analysis for a given model.
    Iteratively builds a KSLossSensitivityAnalysis object and yields an updated
    version after each layer is run. The final result is the complete
    analysis object.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :return: the analysis results for the model with an additional layer at each
        iteration along with a float representing the iteration progress
    """
    model = check_load_model(model)
    prunable = get_prunable_nodes(model)
    analysis = PruningLossSensitivityAnalysis()
    num_layers = len(prunable)

    for index, node in enumerate(prunable):
        node_id = extract_node_id(node)

        yield analysis, KSSensitivityProgress(
            index, node_id, num_layers, float(index) / float(num_layers)
        )

        weight, bias = get_node_params(model, node)
        values = numpy.sort(numpy.abs(weight.val.flatten()))
        prev_index = 0

        for sparsity in sparsity_levels:
            val_index = round(sparsity * values.size)

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

            analysis.add_result(
                node_id, weight.name, index, sparsity, sparse_avg, baseline
            )

    yield analysis, KSSensitivityProgress(num_layers, None, num_layers, 1.0)


def pruning_loss_sens_magnitude(
    model: Union[str, ModelProto],
    sparsity_levels: Union[
        List[float], Tuple[float, ...]
    ] = default_pruning_sparsities_loss(True),
    show_progress: bool = True,
) -> PruningLossSensitivityAnalysis:
    """
    Approximated kernel sparsity (pruning) loss analysis for a given model.
    Returns the results for each prunable param (conv, linear) in the model.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param show_progress: True to log the progress with a tqdm bar, False otherwise
    :return: the analysis results for the model
    """
    analysis = None
    bar = None

    for (analysis, progress) in pruning_loss_sens_magnitude_iter(
        model, sparsity_levels
    ):
        if bar is None and show_progress:
            bar = auto.tqdm(total=progress.total, desc="KS Loss Sensitivity Analysis")

        if bar is not None and progress.val < 1.0:
            bar.update(1)

    if bar is not None:
        bar.close()

    return analysis
