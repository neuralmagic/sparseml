"""
Sensitivity analysis implementations for kernel sparsity on Models against loss funcs.
"""

from typing import Union, List, Tuple, Generator
import logging
from tqdm import auto
import numpy
from onnx import ModelProto
import time

from neuralmagicML.recal import (
    default_check_sparsities_loss,
    default_check_sparsities_perf,
    KSLossSensitivityAnalysis,
    KSPerfSensitivityAnalysis,
    KSSensitivityResult,
)
from neuralmagicML.onnx.utils import (
    get_prunable_nodes,
    extract_node_id,
    get_node_params,
    DataLoader,
    ORTModelRunner,
    NMModelRunner,
    NMBenchmarkModelRunner,
    kl_divergence,
    check_load_model,
)
from neuralmagicML.onnx.recal.mask_ks import prune_model_one_shot


_LOGGER = logging.getLogger(__name__)


__all__ = [
    "approx_ks_loss_sensitivity",
    "one_shot_ks_loss_sensitivity",
    "one_shot_ks_perf_sensitivity",
    "iter_approx_ks_loss_sensitivity",
    "iter_one_shot_ks_loss_sensitivity",
    "iter_one_shot_ks_perf_sensitivity",
    "KSLossSensitivityAnalysis",
    "KSPerfSensitivityAnalysis",
    "KSSensitivityResult",
]


def iter_approx_ks_loss_sensitivity(
    model: Union[str, ModelProto],
    sparsity_levels: Union[
        List[float], Tuple[float, ...]
    ] = default_check_sparsities_loss(True),
) -> Generator[Tuple[KSLossSensitivityAnalysis, float], None, None]:
    """
    Approximated kernel sparsity (pruning) loss analysis for a given model.
    Iteratively builds a KSLossSensitivityAnalysis object and yields an updated
    version after each layer is run. The final result is the complete
    analysis object.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :return: the analysis results for the model with an additional layer at each iteration
        along with a float representing the iteration progress
    """
    model = check_load_model(model)
    prunable = get_prunable_nodes(model)
    analysis = KSLossSensitivityAnalysis()
    num_layers = len(prunable)

    for index, node in enumerate(prunable):
        node_id = extract_node_id(node)
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
        progress = (float(index) + 1) / (float(num_layers) + 1)
        yield analysis, progress


def approx_ks_loss_sensitivity(
    model: Union[str, ModelProto],
    sparsity_levels: Union[
        List[float], Tuple[float, ...]
    ] = default_check_sparsities_loss(True),
) -> KSLossSensitivityAnalysis:
    """
    Approximated kernel sparsity (pruning) loss analysis for a given model.
    Returns the results for each prunable param (conv, linear) in the model.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :return: the analysis results for the model
    """
    analysis = None
    for step in iter_approx_ks_loss_sensitivity(model, sparsity_levels):
        analysis, _ = step
    return analysis


def iter_one_shot_ks_loss_sensitivity(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    steps_per_measurement: int,
    sparsity_levels: List[float] = default_check_sparsities_loss(False),
    show_progress: bool = True,
) -> Generator[Tuple[KSLossSensitivityAnalysis, float], None, None]:
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
    :param show_progress: True to log the progress with a tqdm bar, False otherwise
    :return: the sensitivity results for every node that is prunable, yields update at each
        layer along with iteration progress
    """
    model = check_load_model(model)
    prunable_nodes = get_prunable_nodes(model)
    num_layers = len(prunable_nodes)
    analysis = KSLossSensitivityAnalysis()
    bar = (
        auto.tqdm(
            total=len(prunable_nodes) * len(sparsity_levels) + 1,
            desc="KS Loss Sensitivity Analysis",
        )
        if show_progress
        else None
    )

    runner = (
        ORTModelRunner(model)
        if not NMModelRunner.available()
        else NMModelRunner(model, batch_size)
    )
    _LOGGER.debug("created runner for one shot analysis {}".format(runner))
    base_outputs, _ = runner.run(
        data, desc="", show_progress=False, max_steps=steps_per_measurement,
    )
    _LOGGER.debug("recorded base outputs")
    del runner

    if bar is not None:
        bar.update(1)

    for index, node in enumerate(prunable_nodes):
        node_id = extract_node_id(node)
        weight, bias = get_node_params(model, node)
        _LOGGER.debug("running one shot for node {}".format(node_id))

        for sparsity in sparsity_levels:
            pruned_model = prune_model_one_shot(model, [node], sparsity)
            _LOGGER.debug(
                "created one shot pruned model for sparsity {}".format(sparsity)
            )
            runner = (
                ORTModelRunner(pruned_model)
                if not NMModelRunner.available()
                else NMModelRunner(pruned_model, batch_size)
            )
            _LOGGER.debug("created runner for one shot analysis {}".format(runner))
            pruned_outputs, _ = runner.run(
                data, desc="", show_progress=False, max_steps=steps_per_measurement,
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

            if bar is not None:
                bar.update(1)
        progress = (float(index) + 1) / (float(num_layers) + 1)
        yield analysis, progress

    if bar is not None:
        bar.close()


def one_shot_ks_loss_sensitivity(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    steps_per_measurement: int,
    sparsity_levels: List[float] = default_check_sparsities_loss(False),
    show_progress: bool = True,
) -> KSLossSensitivityAnalysis:
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
    :return: the sensitivity results for every node that is prunable
    """
    analysis = None
    analysis_iter = iter_one_shot_ks_loss_sensitivity(
        model, data, batch_size, steps_per_measurement, sparsity_levels, show_progress
    )
    for step in analysis_iter:
        analysis, _ = step
    return analysis


def iter_one_shot_ks_perf_sensitivity(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    num_cores: int = -1,
    iterations_per_check: int = 10,
    warmup_iterations_per_check: int = 5,
    sparsity_levels: List[float] = default_check_sparsities_perf(),
    show_progress: bool = True,
    wait_between_iters: bool = False,
) -> Generator[Tuple[KSPerfSensitivityAnalysis, float], None, None]:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    Runs a baseline and then sets the sparsity for each layer to a given range
    of values as defined in sparsity_levels to measure their performance for pruning.
    Yields the current KSPerfSensitivityAnalysis after each sparsity level is run.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param data: the data to run through the model
    :param batch_size: the size of the batch to create the model in neural magic for
    :param num_cores: number of physical cores to run on
    :param iterations_per_check: number of iterations to run for perf details
    :param warmup_iterations_per_check: number of iterations to run before perf details
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param show_progress: True to log the progress with a tqdm bar, False otherwise
    :param wait_between_iters: if True, will sleep the thread 0.25s between benchmark
        iterations to allow for other processes to run.
    :return: the sensitivity results for every node that is prunable yields update at each
        layer along with iteration progress
    """
    if not NMBenchmarkModelRunner.available():
        raise ModuleNotFoundError(
            "neuralmagic is not installed on the system, cannot run"
        )

    analysis = KSPerfSensitivityAnalysis(num_cores, batch_size)
    bar = (
        auto.tqdm(total=len(sparsity_levels), desc="KS Perf Sensitivity Analysis")
        if show_progress
        else None
    )
    runner = NMBenchmarkModelRunner(model, batch_size, num_cores)
    _LOGGER.debug("created runner for one shot analysis {}".format(runner))

    for idx, sparsity in enumerate(sparsity_levels):
        if sparsity <= 1e-9:
            # override for the engine which needs None to not impose sparsity
            sparsity = None

        if wait_between_iters:
            time.sleep(0.25)  # hack to release GIL between runs
        results, _ = runner.run(
            data,
            show_progress=False,
            num_iterations=iterations_per_check,
            num_warmup_iterations=warmup_iterations_per_check,
            optimization_level=0,
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

        if bar is not None:
            bar.update(1)
        progress = (float(idx) + 1) / (float(len(sparsity_levels)) + 1)
        yield analysis, progress

    if bar is not None:
        bar.close()


def one_shot_ks_perf_sensitivity(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    num_cores: int = -1,
    iterations_per_check: int = 10,
    warmup_iterations_per_check: int = 5,
    sparsity_levels: List[float] = default_check_sparsities_perf(),
    show_progress: bool = True,
    wait_between_iters: bool = False,
) -> KSPerfSensitivityAnalysis:
    """
    Run a one shot sensitivity analysis for kernel sparsity.
    Runs a baseline and then sets the sparsity for each layer to a given range
    of values as defined in sparsity_levels to measure their performance for pruning.

    :param model: the loaded model or a file path to the onnx model
        to calculate the sparse sensitivity analysis for
    :param data: the data to run through the model
    :param batch_size: the size of the batch to create the model in neural magic for
    :param num_cores: number of physical cores to run on
    :param iterations_per_check: number of iterations to run for perf details
    :param warmup_iterations_per_check: number of iterations to run before perf details
    :param sparsity_levels: the sparsity levels to calculate the loss for for each param
    :param show_progress: True to log the progress with a tqdm bar, False otherwise
    :param wait_between_iters: if True, will sleep the thread 0.25s between benchmark
        iterations to allow for other processes to run.
    :return: the sensitivity results for every node that is prunable
    """
    analysis = None
    analysis_iter = iter_one_shot_ks_perf_sensitivity(
        model,
        data,
        batch_size,
        num_cores,
        iterations_per_check,
        warmup_iterations_per_check,
        sparsity_levels,
        show_progress,
        wait_between_iters,
    )
    for step in analysis_iter:
        analysis, _ = step
    return analysis
