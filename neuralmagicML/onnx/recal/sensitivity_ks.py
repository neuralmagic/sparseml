"""
Sensitivity analysis implementations for kernel sparsity on Models against loss funcs.
"""

from typing import Union, List, Tuple
from tqdm import auto
import numpy
from onnx import ModelProto

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


__all__ = [
    "approx_ks_loss_sensitivity",
    "one_shot_ks_loss_sensitivity",
    "one_shot_ks_perf_sensitivity",
    "KSLossSensitivityAnalysis",
    "KSPerfSensitivityAnalysis",
    "KSSensitivityResult",
]


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
    model = check_load_model(model)
    prunable = get_prunable_nodes(model)
    analysis = KSLossSensitivityAnalysis()

    for index, node in enumerate(prunable):
        node_id = extract_node_id(node)
        weight, bias = get_node_params(model, node)

        values = numpy.sort(numpy.abs(weight.val.flatten()))
        prev_index = None

        for sparsity in sparsity_levels:
            val_index = round(sparsity * values.size)

            if val_index >= len(values):
                val_index = len(values) - 1

            if sparsity <= 1e-9:
                analysis.add_result(
                    node_id, weight.name, index, 0.0, 0.0, baseline=True
                )
            else:
                avg = values[prev_index:val_index].mean().item()
                analysis.add_result(
                    node_id, weight.name, index, sparsity, avg, baseline=False
                )

            prev_index = val_index + 1

    return analysis


def one_shot_ks_loss_sensitivity(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    steps_per_measurement: int,
    sparsity_levels: List[int] = default_check_sparsities_loss(False),
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
    model = check_load_model(model)
    prunable_nodes = get_prunable_nodes(model)
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
    base_outputs, _ = runner.run(
        data, desc="", show_progress=False, max_steps=steps_per_measurement,
    )
    del runner

    if bar is not None:
        bar.update(1)

    for index, node in enumerate(prunable_nodes):
        node_id = extract_node_id(node)
        weight, bias = get_node_params(model, node)

        for sparsity in sparsity_levels:
            pruned_model = prune_model_one_shot(model, [node], sparsity)
            runner = (
                ORTModelRunner(pruned_model)
                if not NMModelRunner.available()
                else NMModelRunner(pruned_model, batch_size)
            )
            pruned_outputs, _ = runner.run(
                data, desc="", show_progress=False, max_steps=steps_per_measurement,
            )
            del runner

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

    if bar is not None:
        bar.close()

    return analysis


def one_shot_ks_perf_sensitivity(
    model: Union[str, ModelProto],
    data: DataLoader,
    batch_size: int,
    num_cores: int = -1,
    iterations_per_check: int = 10,
    warmup_iterations_per_check: int = 5,
    sparsity_levels: List[int] = default_check_sparsities_perf(),
    show_progress: bool = True,
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
    :return: the sensitivity results for every node that is prunable
    """
    if not NMBenchmarkModelRunner.available():
        raise ModuleNotFoundError(
            "neuralmagic is not installed on the system, cannot run"
        )

    analysis = KSPerfSensitivityAnalysis(num_cores, batch_size)
    bar = (
        auto.tqdm(total=len(sparsity_levels), desc="KS Perf Sensitivity Analysis",)
        if show_progress
        else None
    )
    runner = NMBenchmarkModelRunner(model, batch_size, num_cores)

    for sparsity in sparsity_levels:
        if sparsity <= 1e-9:
            # override for the engine which needs None to not impose sparsity
            sparsity = None

        results, _ = runner.run(
            data,
            show_progress=False,
            num_iterations=iterations_per_check,
            num_warmup_iterations=warmup_iterations_per_check,
            optimization_level=0,
            imposed_ks=sparsity,
        )

        for res in results:
            for iter_time in res["iteration_times"]:
                analysis.add_model_result(
                    sparsity if sparsity is not None else 0.0,
                    iter_time,
                    baseline=sparsity is None,
                )

            for index, layer in enumerate(res["layer_info"]):
                analysis.add_result(
                    layer["canonical_name"]
                    if "<none>" not in layer["canonical_name"]
                    else None,
                    layer["name"],
                    index,
                    sparsity if sparsity is not None else layer["kernel_sparsity"],
                    layer["average_run_time_in_ms"],
                    baseline=sparsity is None,
                )

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    return analysis
