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
import time
from typing import Any, Generator, List, NamedTuple, Tuple, Union

import numpy
from onnx import ModelProto
from tqdm import auto

from sparseml.onnx.utils import (
    DataLoader,
    DeepSparseAnalyzeModelRunner,
    DeepSparseModelRunner,
    ORTModelRunner,
    check_load_model,
    extract_node_id,
    get_node_params,
    get_prunable_nodes,
    kl_divergence,
    prune_model_one_shot,
    update_model_param,
)
from sparseml.optim import (
    PruningLossSensitivityAnalysis,
    PruningPerfSensitivityAnalysis,
    PruningSensitivityResult,
    default_pruning_sparsities_loss,
    default_pruning_sparsities_perf,
)
from sparseml.utils import flatten_iterable


_LOGGER = logging.getLogger(__name__)


__all__ = [
    "pruning_loss_sens_approx",
    "pruning_loss_sens_magnitude",
    "pruning_loss_sens_one_shot",
    "pruning_perf_sens_one_shot",
    "pruning_loss_sens_magnitude_iter",
    "pruning_loss_sens_one_shot_iter",
    "pruning_perf_sens_one_shot_iter",
    "PruningLossSensitivityAnalysis",
    "PruningPerfSensitivityAnalysis",
    "PruningSensitivityResult",
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


def pruning_loss_sens_one_shot_iter(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    steps_per_measurement: int,
    sparsity_levels: List[float] = default_pruning_sparsities_loss(False),
    use_deepsparse_inference: bool = False,
) -> Generator[
    Tuple[PruningLossSensitivityAnalysis, KSSensitivityProgress], None, None
]:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    It does not retrain.
    Moves layer by layer to calculate the sensitivity analysis for each and
    resets the previously run layers.
    Updates and yeilds the KSLossSensitivityAnalysis at each layer.
    The loss is calculated by taking the kl_divergence of
    pruned values from the baseline.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param data: the data to run through the model
    :param batch_size: the batch size the data is created for
    :param steps_per_measurement: number of steps (batches) to run through
        the model for each sparsity level on each node
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param use_deepsparse_inference: True to use the DeepSparse inference engine
        to run the analysis, False to use onnxruntime
    :return: the sensitivity results for every node that is prunable,
        yields update at each layer along with iteration progress
    """
    model = check_load_model(model)
    prunable_nodes = get_prunable_nodes(model)
    analysis = PruningLossSensitivityAnalysis()
    num_updates = len(prunable_nodes) * len(sparsity_levels) + 1
    update_num = 0

    yield analysis, KSSensitivityProgress(update_num, None, num_updates, 0.0)

    runner = (
        ORTModelRunner(model)
        if not use_deepsparse_inference
        else DeepSparseModelRunner(model, batch_size)
    )
    _LOGGER.debug("created runner for one shot analysis {}".format(runner))
    base_outputs, _ = runner.run(
        data,
        desc="",
        show_progress=False,
        max_steps=steps_per_measurement,
    )
    _LOGGER.debug("recorded base outputs")
    del runner

    for index, node in enumerate(prunable_nodes):
        node_id = extract_node_id(node)
        weight, bias = get_node_params(model, node)
        _LOGGER.debug("running one shot for node {}".format(node_id))

        for sparsity in sparsity_levels:
            update_num += 1
            yield analysis, KSSensitivityProgress(
                update_num,
                {"node_id": node_id, "sparsity": sparsity},
                num_updates,
                float(update_num) / float(num_updates),
            )

            prune_model_one_shot(model, [node], sparsity)
            _LOGGER.debug(
                "created one shot pruned model for sparsity {}".format(sparsity)
            )
            runner = (
                ORTModelRunner(model)
                if not use_deepsparse_inference
                else DeepSparseModelRunner(model, batch_size)
            )
            _LOGGER.debug("created runner for one shot analysis {}".format(runner))
            pruned_outputs, _ = runner.run(
                data,
                desc="",
                show_progress=False,
                max_steps=steps_per_measurement,
            )
            del runner
            _LOGGER.debug("recorded outputs")

            for base, pruned in zip(base_outputs, pruned_outputs):
                batch_losses = []

                for key, base_array in base.items():
                    pruned_array = pruned[key]
                    loss = kl_divergence(
                        pruned_array,
                        base_array,
                        min(base_array.min(), pruned_array.min()),
                    )
                    batch_losses.append(loss)

                analysis.add_result(
                    node_id,
                    weight.name,
                    index,
                    sparsity,
                    sum(batch_losses),
                    baseline=sparsity < 1e-9,
                )
        # reset node to its baseline density
        update_model_param(model, weight.name, weight.val)

    yield analysis, KSSensitivityProgress(num_updates, None, num_updates, 1.0)


def pruning_loss_sens_one_shot(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    steps_per_measurement: int,
    sparsity_levels: List[float] = default_pruning_sparsities_loss(False),
    show_progress: bool = True,
    use_deepsparse_inference: bool = False,
) -> PruningLossSensitivityAnalysis:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    It does not retrain,.
    Moves layer by layer to calculate the sensitivity analysis for each and
    resets the previously run layers.
    The loss is calculated by taking the kl_divergence of
    pruned values from the baseline.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param data: the data to run through the model
    :param batch_size: the batch size the data is created for
    :param steps_per_measurement: number of steps (batches) to run through
        the model for each sparsity level on each node
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param show_progress: True to log the progress with a tqdm bar, False otherwise
    :param use_deepsparse_inference: True to use the DeepSparse inference engine
        to run the analysis, False to use onnxruntime
    :return: the sensitivity results for every node that is prunable
    """
    analysis = None
    bar = None

    for (analysis, progress) in pruning_loss_sens_one_shot_iter(
        model,
        data,
        batch_size,
        steps_per_measurement,
        sparsity_levels,
        use_deepsparse_inference,
    ):
        if bar is None and show_progress:
            bar = auto.tqdm(total=progress.total, desc="KS Loss Sensitivity Analysis")

        if bar is not None and progress.val < 1.0:
            bar.update(1)

    if bar is not None:
        bar.close()

    return analysis


def pruning_perf_sens_one_shot_iter(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    num_cores: int = None,
    iterations_per_check: int = 10,
    warmup_iterations_per_check: int = 5,
    sparsity_levels: List[float] = default_pruning_sparsities_perf(),
    optimization_level: int = 0,
    iters_sleep_time: float = -1,
) -> Generator[
    Tuple[PruningPerfSensitivityAnalysis, KSSensitivityProgress], None, None
]:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    Runs a baseline and then sets the sparsity for each layer to a given range
    of values as defined in sparsity_levels to measure their performance for pruning.
    Yields the current KSPerfSensitivityAnalysis after each sparsity level is run.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param data: the data to run through the model
    :param batch_size: the size of the batch to create the model in neural magic for
    :param num_cores: number of physical cores to run on. Default is the maximum number
        of cores available
    :param iterations_per_check: number of iterations to run for perf details
    :param warmup_iterations_per_check: number of iterations to run before perf details
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param optimization_level: the optimization level to pass to the DeepSparse
        inference engine for how much to optimize the model.
        Valid values are either 0 for minimal optimizations or 1 for maximal.
    :param iters_sleep_time: the time to sleep the thread between analysis benchmark
        iterations to allow for other processes to run.
    :return: the sensitivity results for every node that is prunable yields update
        at each layer along with iteration progress
    """
    if not DeepSparseAnalyzeModelRunner.available():
        raise ModuleNotFoundError(
            "deepsparse is not installed on the system, cannot run"
        )

    analysis = PruningPerfSensitivityAnalysis(num_cores, batch_size)
    runner = DeepSparseAnalyzeModelRunner(model, batch_size, num_cores)
    _LOGGER.debug("created runner for one shot analysis {}".format(runner))

    for idx, sparsity in enumerate(sparsity_levels):
        if sparsity <= 1e-9:
            # override for the engine which needs None to not impose sparsity
            sparsity = None

        yield analysis, KSSensitivityProgress(
            idx,
            sparsity,
            len(sparsity_levels),
            float(idx) / float(len(sparsity_levels)),
        )

        results, _ = runner.run(
            data,
            show_progress=False,
            num_iterations=iterations_per_check,
            num_warmup_iterations=warmup_iterations_per_check,
            optimization_level=optimization_level,
            imposed_ks=sparsity,
        )
        _LOGGER.debug("measured results for one shot sparsity {}".format(sparsity))

        for res in results:
            for iter_time in res["iteration_times"]:
                analysis.add_model_result(
                    sparsity if sparsity is not None else 0.0,
                    iter_time / 1000.0,
                    baseline=sparsity is None,
                )

            for index, layer in enumerate(res["layer_info"]):
                analysis.add_result(
                    layer["canonical_name"],
                    layer["name"],
                    index,
                    sparsity if sparsity is not None else layer["kernel_sparsity"],
                    layer["average_run_time_in_ms"] / 1000.0,
                    baseline=sparsity is None,
                )

        if iters_sleep_time >= 0.0:
            time.sleep(iters_sleep_time)  # hack to release GIL between runs

    yield analysis, KSSensitivityProgress(
        len(sparsity_levels),
        None,
        len(sparsity_levels),
        1.0,
    )


def pruning_perf_sens_one_shot(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    num_cores: int = None,
    iterations_per_check: int = 10,
    warmup_iterations_per_check: int = 5,
    sparsity_levels: List[float] = default_pruning_sparsities_perf(),
    show_progress: bool = True,
    wait_between_iters: bool = False,
) -> PruningPerfSensitivityAnalysis:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    Runs a baseline and then sets the sparsity for each layer to a given range
    of values as defined in sparsity_levels to measure their performance for pruning.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param data: the data to run through the model
    :param batch_size: the size of the batch to create the model in neural magic for
    :param num_cores: number of physical cores to run on. Default is the maximum
        available
    :param iterations_per_check: number of iterations to run for perf details
    :param warmup_iterations_per_check: number of iterations to run before perf details
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param show_progress: True to log the progress with a tqdm bar, False otherwise
    :param wait_between_iters: if True, will sleep the thread 0.25s between analysis
        benchmark iterations to allow for other processes to run.
    :return: the sensitivity results for every node that is prunable
    """
    analysis = None
    bar = None

    for (analysis, progress) in pruning_perf_sens_one_shot_iter(
        model,
        data,
        batch_size,
        num_cores,
        iterations_per_check,
        warmup_iterations_per_check,
        sparsity_levels,
        wait_between_iters,
    ):
        if bar is None and show_progress:
            bar = auto.tqdm(total=progress.total, desc="KS Perf Sensitivity Analysis")

        if bar is not None and progress.val < 1.0:
            bar.update(1)

    if bar is not None:
        bar.close()

    return analysis
