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
Sensitivity analysis implementations for learning rate on Modules against loss funcs.
"""

from typing import Any, Callable, List, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sparseml.optim import LRLossSensitivityAnalysis
from sparseml.pytorch.utils import (
    DEFAULT_LOSS_KEY,
    BaseLogger,
    LossWrapper,
    ModuleRunFuncs,
    ModuleRunResults,
    ModuleTrainer,
    infinite_data_loader,
    set_optim_learning_rate,
)


__all__ = ["default_exponential_check_lrs", "lr_loss_sensitivity"]


def default_exponential_check_lrs(
    init_lr: float = 1e-6, final_lr: float = 0.5, lr_mult: float = 1.1
) -> Tuple[float, ...]:
    """
    Get the default learning rates to check between init_lr and final_lr.

    :param init_lr: the initial learning rate in the returned list
    :param final_lr: the final learning rate in the returned list
    :param lr_mult: the multiplier increase for each step between
        init_lr and final_lr
    :return: the list of created lrs that increase exponentially between
        init_lr and final_lr according to lr_mult
    """
    check_lrs = [init_lr]  # type: List[float]

    while check_lrs[-1] < final_lr:
        check_lrs.append(check_lrs[-1] * lr_mult)

    check_lrs.append(final_lr)

    return tuple(check_lrs)


def _sensitivity_callback(
    check_lrs: Union[List[float], Tuple[float, ...]],
    steps_per_measurement: int,
    optim: Optimizer,
    analysis: LRLossSensitivityAnalysis,
    loss_key: str,
) -> Tuple[Callable, Callable]:
    measurement_steps = 0
    check_index = -1
    lr_results = None

    def complete_lr():
        nonlocal measurement_steps
        nonlocal check_index
        nonlocal lr_results

        if measurement_steps > 0 and check_index >= 0 and check_index < len(check_lrs):
            lr_res = [res.item() for res in lr_results.result_list_tensor(loss_key)]
            analysis.add_result(check_lrs[check_index], lr_res)

        measurement_steps = 0
        check_index += 1
        lr_results = ModuleRunResults()

        if check_index < len(check_lrs):
            set_optim_learning_rate(optim, check_lrs[check_index])

    complete_lr()  # initial to set the lr

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

        if measurement_steps >= steps_per_measurement:
            complete_lr()

        lr_results.append(losses, batch_size)

    def completed():
        complete_lr()  # make sure we didn't miss any

    return batch_end, completed


def lr_loss_sensitivity(
    module: Module,
    data: DataLoader,
    loss: Union[LossWrapper, Callable[[Any, Any], Tensor]],
    optim: Optimizer,
    device: str,
    steps_per_measurement: int,
    check_lrs: Union[List[float], Tuple[float, ...]] = default_exponential_check_lrs(),
    loss_key: str = DEFAULT_LOSS_KEY,
    trainer_run_funcs: ModuleRunFuncs = None,
    trainer_loggers: List[BaseLogger] = None,
    show_progress: bool = True,
) -> LRLossSensitivityAnalysis:
    """
    Implementation for handling running sensitivity analysis for
    learning rates on modules.

    :param module: the module to run the learning rate sensitivity analysis over,
        it is expected to already be on the correct device
    :param data: the data to run through the module for calculating
        the sensitivity analysis
    :param loss: the loss function to use for the sensitivity analysis
    :param optim: the optimizer to run the sensitivity analysis with
    :param device: the device to run the analysis on; ex: cpu, cuda.
        module must already be on that device, this is used to place then data
        on that same device.
    :param steps_per_measurement: the number of batches to run through for
        the analysis at each LR
    :param check_lrs: the learning rates to check for analysis
        (will sort them small to large before running)
    :param loss_key: the key for the loss function to track in the returned dict
    :param trainer_run_funcs: override functions for ModuleTrainer class
    :param trainer_loggers: loggers to log data to while running the analysis
    :param show_progress: track progress of the runs if True
    :return: a list of tuples containing the analyzed learning rate at 0
        and the ModuleRunResults in 1, ModuleRunResults being a collection
        of all the batch results run through the module at that LR
    """
    analysis = LRLossSensitivityAnalysis()
    trainer = ModuleTrainer(
        module,
        device,
        loss,
        optim,
        loggers=trainer_loggers,
        log_summary=False,
        log_steps=max(1, round(steps_per_measurement / 10)),
    )
    batch_end, completed = _sensitivity_callback(
        check_lrs, steps_per_measurement, optim, analysis, loss_key
    )
    batch_end_hook = trainer.run_hooks.register_batch_end_hook(batch_end)
    if trainer_run_funcs is not None:
        trainer.run_funcs.copy(trainer_run_funcs)

    data_loader = infinite_data_loader(data)
    trainer.run(
        data_loader,
        desc="LR Analysis",
        show_progress=show_progress,
        track_results=False,
        max_steps=steps_per_measurement * len(check_lrs),
    )
    completed()
    batch_end_hook.remove()

    return analysis
