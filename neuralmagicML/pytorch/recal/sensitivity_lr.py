"""
Sensitivity analysis implementations for learning rate on Modules against loss funcs.
"""

from typing import List, Tuple, Callable, Dict, Any, Union
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import pandas

from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle

from neuralmagicML.recal import LRLossSensitivityProgress
from neuralmagicML.utils import (
    clean_path,
    create_parent_dirs,
)
from neuralmagicML.pytorch.utils import (
    LossWrapper,
    DEFAULT_LOSS_KEY,
    ModuleTrainer,
    ModuleRunFuncs,
    ModuleRunResults,
    model_to_device,
)


__all__ = ["default_exponential_check_lrs", "lr_loss_sensitivity"]


def _measured_data_loader(data_loader: DataLoader, num_yields: int):
    counter = 0
    while True:
        for data in data_loader:
            yield data

            counter += 1
            if counter == num_yields:
                return


def default_exponential_check_lrs(
    init_lr: float = 1e-9, final_lr: float = 1e0, lr_mult: float = 1.1
):
    """
    Get the default learning rates to check between init_lr and final_lr.

    :param init_lr: the initial learning rate in the returned list
    :param final_lr: the final learning rate in the returned list
    :param lr_mult: the multiplier increase for each step between
        init_lr and final_lr
    :return: the list of created lrs that increase exponentially between
        init_lr and final_lr according to lr_mult
    """
    check_lrs = [init_lr]

    while check_lrs[-1] < final_lr:
        check_lrs.append(check_lrs[-1] * lr_mult)

    check_lrs.append(final_lr)

    return check_lrs


def lr_loss_sensitivity(
    module: Module,
    data: Dataset,
    loss_fn: LossWrapper,
    device: str,
    batch_size: int,
    batches_per_measurement: int,
    check_lrs: List[float],
    loss_key: str = DEFAULT_LOSS_KEY,
    sgd_args: Dict = None,
    loader_args: Dict = None,
    data_loader_const: Callable = DataLoader,
    trainer_run_funcs: ModuleRunFuncs = None,
    progress_hook: Union[Callable, None] = None,
) -> List[Tuple[float, float]]:
    """
    Implementation for handling running sensitivity analysis for
    learning rates on modules.

    :param module: the module to run the learning rate sensitivity analysis over
    :param data: the data to run through the module for calculating
        the sensitivity analysis
    :param loss_fn: the loss function to use for the sensitivity analysis
    :param device: the device to run the analysis on; ex: cpu, cuda, cuda:0,1
    :param batch_size: the batch size to run through them model for the analysis
    :param batches_per_measurement: the number of batches to run through for
        the analysis at each LR
    :param check_lrs: the learning rates to check for analysis
        (will sort them small to large before running)
    :param loss_key: the key for the loss function to track in the returned dict
        while running
    :param sgd_args: any args to add to the SGD optimizer that
        will be created for analysis
    :param loader_args: any args to add to the DataLoader
    :param data_loader_const: a data loader constructor to create
        the data loader with, default is DataLoader
    :param trainer_run_funcs: override functions for ModuleTrainer class
    :param progress_hook: a hook to handle reporting progress updates to
    :return: a list of tuples containing the analyzed learning rate at 0
        and the ModuleRunResults in 1, ModuleRunResults being a collection
        of all the batch results run through the module at that LR
    """
    if loader_args is None:
        loader_args = {}

    if sgd_args is None:
        sgd_args = {}

    module, device, device_ids = model_to_device(module, device)
    device_str = (
        device
        if device_ids is None or len(device_ids) < 2
        else "{}:{}".format(device, device_ids[0])
    )

    check_lrs = sorted(check_lrs)
    optim = SGD(module.parameters(), lr=1.0, **sgd_args)
    results = []  # type: List[Tuple[float, ModuleRunResults]]

    progress = LRLossSensitivityProgress(
        lr_index=-1,
        lr=-1,
        check_lrs=check_lrs,
        batch=-1,
        batches_per_measurement=batches_per_measurement,
    )

    def _batch_end(
        _epoch: int, _step: int, _batch_size: int, _data: Any, _pred: Any, _losses: Any,
    ):
        if progress.lr_index != -1:
            progress.batch += 1
            results[-1][1].append(_losses, _batch_size)

        if progress.lr_index + 1 == len(progress.check_lrs):
            return

        if (
            progress.batch == -1
            or progress.batch + 1 == progress.batches_per_measurement
        ):
            progress.batch = 0
            progress.lr_index += 1
            progress.lr = progress.check_lrs[progress.lr_index]
            results.append((progress.lr, ModuleRunResults()))

            for param_group in optim.param_groups:
                param_group["lr"] = progress.lr

        if progress_hook:
            progress_hook(progress)

    trainer = ModuleTrainer(module, device_str, loss_fn, optim)
    batch_end_hook = trainer.run_hooks.register_batch_end_hook(_batch_end)
    if trainer_run_funcs is not None:
        trainer.run_funcs.copy(trainer_run_funcs)

    data_loader = data_loader_const(data, batch_size=batch_size, **loader_args)
    data_loader = _measured_data_loader(
        data_loader, len(check_lrs) * batches_per_measurement
    )
    trainer.run(data_loader, desc="")
    batch_end_hook.remove()

    if progress_hook:
        progress.check_lrs = len(progress.check_lrs) - 1
        progress.batch = progress.batches_per_measurement - 1
        progress_hook(progress)

    return [(lr, run.result_mean(loss_key).item()) for (lr, run) in results]
