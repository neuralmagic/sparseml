"""
code related to calculating kernel sparsity sensitivity analysis for models
"""

from typing import List, Callable, Dict, Any, Union

from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from neuralmagicML.recal import (
    KSLossSensitivityProgress,
    KSLossSensitivityResult,
    KSLossSensitivityAnalysis,
)
from neuralmagicML.pytorch.datasets import EarlyStopDataset, CacheableDataset
from neuralmagicML.pytorch.utils import (
    ModuleTester,
    LossWrapper,
    DEFAULT_LOSS_KEY,
    ModuleRunFuncs,
    get_prunable_layers,
    model_to_device,
)
from neuralmagicML.pytorch.utils.module_analyzer import ModuleAnalyzer
from neuralmagicML.pytorch.recal.kernel.mask import ModuleParamKSMask


__all__ = [
    "one_shot_ks_loss_sensitivity",
]


def one_shot_ks_loss_sensitivity(
    module: Module,
    data: Dataset,
    loss_fn: LossWrapper,
    device: str,
    batch_size: int,
    samples_per_measurement: int,
    sparsity_levels: List[int] = None,
    cache_data: bool = True,
    loader_args: Dict = None,
    data_loader_const: Callable = DataLoader,
    tester_run_funcs: ModuleRunFuncs = None,
    progress_hook: Union[Callable, None] = None,
) -> KSLossSensitivityAnalysis:
    """
    Run a one shot sensitivity analysis for kernel sparsity
    It does not retrain, and instead puts the model to eval mode.
    Moves layer by layer to calculate the sensitivity analysis for each and resets the previously run layers.
    Note, by default it caches the data.
    This means it is not parallel for data loading and the first run can take longer.
    Subsequent sparsity checks for layers and levels will be much faster.

    :param module: the module to run the kernel sparsity sensitivity analysis over
                       will extract all prunable layers out
    :param data: the data to run through the module for calculating the sensitivity analysis
    :param loss_fn: the loss function to use for the sensitivity analysis
    :param device: the device to run the analysis on; ex: cpu, cuda, cuda:0,1
    :param batch_size: the batch size to run through the model in eval mode
    :param samples_per_measurement: the number of samples or items to take for each measurement at each sparsity lev
    :param sparsity_levels: the sparsity levels to check for each layer to calculate sensitivity
    :param cache_data: True to cache the data in CPU RAM instead of reloading from disk
                       False to not cache. Note, num_workers must be 0 for this to work, so first load is slow
    :param loader_args: the arguments to supply to the DataLoader other than the dataset and batch size
    :param data_loader_const: the constructor used to create a DataLoader instance, defaults to the regular pytorch
    :param tester_run_funcs: override functions to use in the ModuleTester that runs
    :param progress_hook: a hook to handle reporting progress updates to
    :return: the sensitivity results for every layer that is prunable
    """
    data = (
        EarlyStopDataset(data, samples_per_measurement)
        if len(data) > samples_per_measurement > 0
        else data
    )

    if loader_args is None:
        loader_args = {}

    if cache_data:
        # cacheable dataset does not work with parallel data loaders
        if "num_workers" in loader_args and loader_args["num_workers"] != 0:
            raise ValueError("num_workers must be 0 for dataset cache")

        loader_args["num_workers"] = 0
        data = CacheableDataset(data)

    if sparsity_levels is None:
        sparsity_levels = [0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]

    if progress_hook is None:
        progress_hook = KSLossSensitivityProgress.standard_update_hook()

    module, device, device_ids = model_to_device(module, device)
    analyzer = ModuleAnalyzer(module, enabled=True)
    device_str = (
        device
        if device_ids is None or len(device_ids) < 2
        else "{}:{}".format(device, device_ids[0])
    )

    layers = get_prunable_layers(module)
    progress = KSLossSensitivityProgress(
        layer_index=-1,
        layer_name="",
        layers=[lay[0] for lay in layers],
        sparsity_index=-1,
        sparsity_levels=sparsity_levels,
        measurement_step=-1,
        samples_per_measurement=samples_per_measurement,
    )

    def _batch_end(
        _epoch: int, _step: int, _batch_size: int, _data: Any, _pred: Any, _losses: Any,
    ):
        progress.measurement_step = _step
        analyzer.enabled = False
        if progress_hook:
            progress_hook(progress)

    tester = ModuleTester(module, device_str, loss_fn)
    batch_end_hook = tester.run_hooks.register_batch_end_hook(_batch_end)
    if tester_run_funcs is not None:
        tester.run_funcs.copy(tester_run_funcs)

    analysis = KSLossSensitivityAnalysis()

    for layer_index, (name, layer) in enumerate(layers):
        progress.layer_index = layer_index
        progress.layer_name = name
        progress.sparsity_index = -1
        progress.measurement_step = -1
        if progress_hook:
            progress_hook(progress)

        sparsities_loss = []
        mask = ModuleParamKSMask(layer, store_init=True)
        mask.enabled = True

        for sparsity_index, sparsity_level in enumerate(sparsity_levels):
            progress.sparsity_index = sparsity_index
            progress.measurement_step = -1
            if progress_hook:
                progress_hook(progress)

            mask.set_param_mask_from_sparsity(sparsity_level)
            data_loader = data_loader_const(data, batch_size, **loader_args)
            res = tester.run(
                data_loader, desc="", show_progress=False, track_results=True
            )
            sparsities_loss.append(
                (sparsity_level, res.result_mean(DEFAULT_LOSS_KEY).item())
            )

        mask.enabled = False
        mask.reset()
        del mask

        desc = analyzer.layer_desc(name)
        analysis.results.append(
            KSLossSensitivityResult(name, "weight", desc.type_, sparsities_loss)
        )

    batch_end_hook.remove()

    return analysis
