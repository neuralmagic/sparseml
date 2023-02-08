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
Code related to running a module through training and testing over a dataset.
Allows reporting of progress and override functions and hooks.
"""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import auto

from sparseml.pytorch.utils.helpers import (
    get_optim_learning_rate,
    tensors_batch_size,
    tensors_module_forward,
    tensors_to_device,
)
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.utils.loss import DEFAULT_LOSS_KEY, LossWrapper


try:
    from torch.cuda.amp import GradScaler, autocast

    amp_import_error = None
except Exception as amp_error:
    autocast = None
    GradScaler = None
    amp_import_error = amp_error


__all__ = [
    "def_model_backward",
    "ModuleRunFuncs",
    "ModuleRunHooks",
    "ModuleRunResults",
    "ModuleDeviceContext",
    "ModuleTester",
    "ModuleTrainer",
]


def def_model_backward(
    losses: Dict[str, Tensor], model: Module, scaler: GradScaler = None
):
    """
    Default function to perform a backwards pass for a model and the calculated losses
    Calls backwards for the DEFAULT_LOSS_KEY in losses Dict

    :param model: the model to run the backward for
    :param losses: the losses dictionary containing named tensors,
                   DEFAULT_LOSS_KEY is expected to exist and backwards is called on that
    :param scaler: GradScaler object for running in mixed precision with amp. If scaler
        is not None will call scaler.scale on the loss object. Default is None
    """
    # assume loss is at default loss key
    loss = losses[DEFAULT_LOSS_KEY]
    if scaler is not None:
        loss = scaler.scale(loss)
    loss.backward()


class ModuleRunHooks(object):
    """
    Container for hooks that can be added to module runs like training and testing
    for different stages of running a batch through a model.

    | Lifecycle:
    |   - data batch size callback
    |   - data to device callback
    |   - batch start hook
    |   - data model forward callback
    |   - batch forward hook
    |   - loss calculation
    |   - batch loss hook
    |   - model backward callback
    |   - batch backward hook
    |   - optimizer / gradient update
    |   - batch end hook
    """

    def __init__(self):
        self._batch_start_hooks = OrderedDict()
        self._batch_forward_hooks = OrderedDict()
        self._batch_loss_hooks = OrderedDict()
        self._batch_backward_hooks = OrderedDict()
        self._batch_end_hooks = OrderedDict()

    def register_batch_start_hook(
        self, hook: Callable[[int, int, int, Any], None]
    ) -> RemovableHandle:
        """
        Called at the start of a batch with the following info:
        (counter, step_count, batch_size, data)
        where counter is passed in to the run (ex: epoch),
        step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch,
        data is the data output from the loader

        :param hook: the hook to add that is called into when reached in the
            batch process
        :return: a removable handle to remove the hook when desired
        """
        handle = RemovableHandle(self._batch_start_hooks)
        self._batch_start_hooks[handle.id] = hook

        return handle

    def register_batch_forward_hook(
        self, hook: Callable[[int, int, int, Any, Any], None]
    ) -> RemovableHandle:
        """
        Called after forward execution of a batch in the model with the following info:
        (counter, step_count, batch_size, data, pred)
        where counter is passed in to the run (ex: epoch),
        step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch,
        data is the data output from the loader,
        pred is the result from the model after the forward

        :param hook: the hook to add that is called into when reached in the
            batch process
        :return: a removable handle to remove the hook when desired
        """
        handle = RemovableHandle(self._batch_forward_hooks)
        self._batch_forward_hooks[handle.id] = hook

        return handle

    def register_batch_loss_hook(
        self, hook: Callable[[int, int, int, Any, Any, Dict[str, Tensor]], None]
    ):
        """
        Called after loss calculation of the batch with the following info:
        (counter, step_count, batch_size, data, pred, losses)
        where counter is passed in to the run (ex: epoch),
        step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch,
        data is the data output from the loader,
        pred is the result from the model after the forward,
        losses are the resulting loss dictionary

        :param hook: the hook to add that is called into when reached in the
            batch process
        :return: a removable handle to remove the hook when desired
        """
        handle = RemovableHandle(self._batch_loss_hooks)
        self._batch_loss_hooks[handle.id] = hook

        return handle

    def register_batch_backward_hook(
        self, hook: Callable[[int, int, int, Any, Any, Dict[str, Tensor]], None]
    ):
        """
        Called after calling backward on the loss for the batch with the following info:
        (counter, step_count, batch_size, data, pred, losses)
        where counter is passed in to the run (ex: epoch),
        step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch,
        data is the data output from the loader,
        pred is the result from the model after the forward,
        losses are the resulting loss dictionary

        :param hook: the hook to add that is called into when reached in the
            batch process
        :return: a removable handle to remove the hook when desired
        """
        handle = RemovableHandle(self._batch_backward_hooks)
        self._batch_backward_hooks[handle.id] = hook

        return handle

    def register_batch_end_hook(
        self, hook: Callable[[int, int, int, Any, Any, Dict[str, Tensor]], None]
    ):
        """
        Called after all calculations are done for the batch with the following info:
        (counter, step_count, batch_size, data, pred, losses)
        where counter is passed in to the run (ex: epoch),
        step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch,
        data is the data output from the loader,
        pred is the result from the model after the forward,
        losses are the resulting loss dictionary

        :param hook: the hook to add that is called into when reached in the
            batch process
        :return: a removable handle to remove the hook when desired
        """
        handle = RemovableHandle(self._batch_end_hooks)
        self._batch_end_hooks[handle.id] = hook

        return handle

    def invoke_batch_start(
        self, counter: int, step_count: int, batch_size: int, data: Any
    ):
        for hook in self._batch_start_hooks.values():
            hook(counter, step_count, batch_size, data)

    def invoke_batch_forward(
        self, counter: int, step_count: int, batch_size: int, data: Any, pred: Any
    ):
        for hook in self._batch_forward_hooks.values():
            hook(counter, step_count, batch_size, data, pred)

    def invoke_batch_loss(
        self,
        counter: int,
        step_count: int,
        batch_size: int,
        data: Any,
        pred: Any,
        losses: Dict[str, Tensor],
    ):
        for hook in self._batch_loss_hooks.values():
            hook(counter, step_count, batch_size, data, pred, losses)

    def invoke_batch_backward(
        self,
        counter: int,
        step_count: int,
        batch_size: int,
        data: Any,
        pred: Any,
        losses: Dict[str, Tensor],
    ):
        for hook in self._batch_backward_hooks.values():
            hook(counter, step_count, batch_size, data, pred, losses)

    def invoke_batch_end(
        self,
        counter: int,
        step_count: int,
        batch_size: int,
        data: Any,
        pred: Any,
        losses: Dict[str, Tensor],
    ):
        for hook in self._batch_end_hooks.values():
            hook(counter, step_count, batch_size, data, pred, losses)


class ModuleRunFuncs(object):
    """
    Functions used as callables to calculate or perform necessary operations
    for running a model through training or testing.

    | Lifecycle:
    |   - data batch size callback
    |   - data to device callback
    |   - batch start hook
    |   - data model forward callback
    |   - batch forward hook
    |   - loss calculation
    |   - batch loss hook
    |   - model backward callback
    |   - batch backward hook
    |   - optimizer / gradient update
    |   - batch end hook
    """

    def __init__(self):
        self._batch_size = tensors_batch_size
        self._to_device = tensors_to_device
        self._model_forward = tensors_module_forward
        self._model_backward = def_model_backward

    @property
    def batch_size(self) -> Callable[[Any], int]:
        """
        :return used to calculate the batch size of a given grouping of tensors.
            Expected to be called with the output from a data loader and
            then return an int representing the batch size.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: Callable[[Any], int]):
        """
        Used to calculate the batch size of a given grouping of tensors.
        Expected to be called with the output from a data loader
        then return an int representing the batch size.

        :param value: the callable used to calculate batch size
            for a grouping of tensors
        """
        self._batch_size = value

    @property
    def to_device(self) -> Callable[[Any, str], Any]:
        """
        :return used to place a given grouping of tensors onto the proper device.
            Expected to be called with the output from a data loader and the
            desired device as a string then return the grouping on the proper device.
        """
        return self._to_device

    @to_device.setter
    def to_device(self, value: Callable[[Any, str], Any]):
        """
        Used to place a given grouping of tensors onto the proper device.
        Expected to be called with the output from a data loader and the
        desired device as a string then return the grouping on the proper device

        :param value: the callable used to place a grouping of tensors onto
            the proper device
        """
        self._to_device = value

    @property
    def model_forward(self) -> Callable[[Any, Module], Any]:
        """
        :return used to propagate a given grouping of tensors through a model and
            return the result.
            Expected to be called with the model and the output from a data loader
            then return the result from the model forward pass.
        """
        return self._model_forward

    @model_forward.setter
    def model_forward(self, value: Callable[[Any, Module], Any]):
        """
        Used to propagate a given grouping of tensors through a model
        and return the result.
        Expected to be called with the model and the output from a data loader
        then return the result from the model forward pass.

        :param value: the callable used to run a grouping of tensors through a model
        :return: the result of running the data through the model
        """
        self._model_forward = value

    @property
    def model_backward(self) -> Callable[[Dict[str, Tensor], Module], None]:
        """
        :return used to call backward for a given model and the calculated losses.
            Expected to be called with the model and the output from the loss function
            as a dict mapping of names to tensors returns nothing
        """
        return self._model_backward

    @model_backward.setter
    def model_backward(self, value: Callable[[Dict[str, Tensor], Module], None]):
        """
        Used to call backward for a given model and the calculated losses.
        Expected to be called with the model and the output from the loss function
        as a dict mapping of names to tensors returns nothing

        :param value: the callable used to run a backwards pass for the given
            loss functions
        """
        self._model_backward = value

    def copy(self, run_funcs):
        """
        Copy the functions from the current instance into a new instance

        :param run_funcs: the instance to copy the functions into
        """
        run_funcs = run_funcs  # type: ModuleRunFuncs

        self._batch_size = run_funcs._batch_size
        self._to_device = run_funcs._to_device
        self._model_forward = run_funcs._model_forward
        self._model_backward = run_funcs._model_backward


class ModuleRunResults(object):
    """
    Class containing the results / losses from a model run for training or testing
    Keeps all result values as a dictionary and Tensor containing all values
    """

    def __init__(self):
        self._results = {}

    def __repr__(self):
        results = [
            "{}={}".format(key, self.result_mean(key).item()) for key in self._results
        ]

        return "ModuleRunResults({})".format(", ".join(results))

    @property
    def results(self) -> Dict[str, List[Tensor]]:
        """
        All of the stored results for the loss functions

        :return: a dictionary containing a mapping of name (str) to a list of tensors
            that were recorded for that loss
        """
        return self._results

    def result(self, key: str) -> List[Tensor]:
        """
        The result of a single loss function

        :param key: the name of the loss function to get the results for
        :return: a list of tensors containing all of the results for that loss
        """
        return self._results[key]

    def result_list_tensor(self, key: str) -> Tensor:
        """
        Get the results as a list tensor where all items have been stacked into
        the first index of the tensor.

        :param key: the name of the loss function to get the results for
        :return: a tensor containing all of the tensors for that result
        """
        res = self.result(key)

        return torch.cat(res)

    def result_mean(self, key: str) -> Tensor:
        """
        The mean result of a single loss function

        :param key: the name of the loss function to get the mean result for
        :return: a single tensor containing the average of all the results for that loss
        """
        res = self.result_list_tensor(key)

        return torch.mean(res)

    def result_std(self, key: str) -> Tensor:
        """
        The standard deviation of the result for a single loss function

        :param key: the name of the loss function to get the standard
            deviation result for
        :return: a single tensor containing the standard deviation of all
            the results for that loss
        """
        res = self.result_list_tensor(key)

        return torch.std(res)

    def append(self, losses: Dict[str, Tensor], batch_size: int):
        """
        add new losses to the current stored results

        :param losses: the losses to be added
        :param batch_size: the batch size the losses were run for
        """
        for key, val in losses.items():
            if key not in self._results:
                self._results[key] = []

            result = val.detach_().cpu()
            result = result.repeat(batch_size)
            self._results[key].append(result)


class ModuleDeviceContext(object):
    """
    Simple class to define device settings or context to be used when running a Module

    :param use_mixed_precision: set True to execute model using mixed precision with
        torch.cuda.amp. Default is False
    :param world_size: the world size (total number of devices) used when running
        the given module using DistributedDataParallel. Losses will be scaled by the
        world size. Default is 1.
    """

    def __init__(self, use_mixed_precision: bool = False, world_size: int = 1):
        self._use_mixed_precision = use_mixed_precision
        self._world_size = world_size
        self._validate()

    @staticmethod
    def default_context():
        """
        :return: A ModuleDeviceContext with default settings enabled
        """
        return ModuleDeviceContext(use_mixed_precision=False, world_size=1)

    @property
    def use_mixed_precision(self) -> bool:
        """
        :return: True if mixed precision with torch.cuda.amp should be used.
            False otherwise
        """
        return self._use_mixed_precision

    @use_mixed_precision.setter
    def use_mixed_precision(self, value: bool):
        """
        :param value: True if mixed precision with torch.cuda.amp should be used.
            False otherwise
        """
        self._use_mixed_precision = value
        self._validate()

    @property
    def world_size(self) -> int:
        """
        :return: the world size (total number of devices) used when running
        the given module using DistributedDataParallel. Losses will be scaled by the
        world size
        """
        return self._world_size

    @world_size.setter
    def world_size(self, value: int):
        """
        :param value: the world size (total number of devices) used when running
        the given module using DistributedDataParallel. Losses will be scaled by the
        world size
        """
        self._world_size = value
        self._validate()

    def _validate(self):
        assert isinstance(
            self.use_mixed_precision, bool
        ), "use_mixed_precision must be a boolean"
        assert (
            isinstance(self.world_size, int) and self.world_size > 0
        ), "world_size must be a positive int"


class ModuleRunner(ABC):
    """
    Abstract class for running data through a module and recording the results

    :param module: the model to run evaluation for
    :param device: the default device to run evaluation on
        (where data will be copied to)
    :param loss: the loss functions callable used to calculate loss values after
        executing a forward pass
    :param loggers: Optional list of loggers to log the modification process to
    :param log_name: the key to store all log files under
    :param log_steps: The number of steps (batches) to log at,
        ex 100 will log every 100 batches
    :param log_summary: True to log the final summary results after the run completes
    :param device_context: ModuleDeviceContext with settings to enable mixed precision
        using torch.cuda.amp or adjust losses when using DistributedDataParallel.
        Default settings do not use mixed precision or account for DDP.
    """

    def __init__(
        self,
        module: Module,
        device: str,
        loss: Union[LossWrapper, Callable[[Any, Any], Tensor]],
        loggers: Optional[List[BaseLogger]],
        log_name: str,
        log_steps: int,
        log_summary: bool,
        device_context: ModuleDeviceContext = ModuleDeviceContext.default_context(),
    ):
        self._module = module
        self._device = device
        self._loss = (
            loss
            if isinstance(loss, LossWrapper)
            else LossWrapper(loss, deconstruct_tensors=False)
        )
        self._loggers = loggers
        self._log_name = log_name
        self._log_steps = log_steps
        self._log_summary = log_summary
        self._device_context = device_context

        self._run_funcs = ModuleRunFuncs()
        self._run_hooks = ModuleRunHooks()

    @property
    def module(self) -> Module:
        """
        :return: the model to run
        """
        return self._module

    @property
    def device(self) -> str:
        """
        :return: the default device to run on (where data will be copied to)
        """
        return self._device

    @property
    def loss(self) -> LossWrapper:
        """
        :return: the loss functions callable used to calculate loss values
            after executing a forward pass
        """
        return self._loss

    @property
    def run_funcs(self) -> ModuleRunFuncs:
        """
        :return: functions used while running evaluation of the model as
            callbacks to do certain stages
        """
        return self._run_funcs

    @property
    def run_hooks(self) -> ModuleRunHooks:
        """
        :return: hooks used while running evaluation of the model to
            receive intermediate results
        """
        return self._run_hooks

    @property
    def device_context(self) -> ModuleDeviceContext:
        """
        :return: ModuleDeviceContext with settings for enabling mixed precision
        using torch.cuda.amp or adjusting losses when using DistributedDataParallel.
        """
        return self._device_context

    def run(
        self,
        data_loader: DataLoader,
        desc: str,
        counter: int = -1,
        show_progress: bool = True,
        track_results: bool = True,
        max_steps: int = -1,
    ) -> Union[None, ModuleRunResults]:
        """
        Run evaluation over all the data in the given data loader

        :param data_loader: the data loader used to gather batches to be
            run through the model
        :param desc: description used in the progress indicator
        :param counter: counter passed to the hooks for external
            state keeping (ex: epoch)
        :param show_progress: True to show a progress bar, False otherwise
        :param track_results: True to track and return the results of the evaluation,
            False to return None
        :param max_steps: maximum number of steps/batches to run through,
            will stop after reaching this. if <= 0 then no restriction is placed
        :return: the results of evaluation if track_results else None
        """
        if self._log_summary and not track_results:
            raise ValueError(
                "runner must be run with track_results=True to log the final results"
            )

        self._runner_setup()

        try:
            counter_len = len(data_loader)
        except Exception:
            # can't track data loaders length
            counter_len = 0

        if max_steps > 0 and counter_len > 0:
            progress_steps = min(max_steps, counter_len)
        elif max_steps > 0:
            progress_steps = max_steps
        elif counter_len > 0:
            progress_steps = counter_len
        else:
            progress_steps = None

        data_iter = (
            enumerate(data_loader)
            if not show_progress
            else enumerate(auto.tqdm(data_loader, desc=desc, total=progress_steps))
        )
        results = ModuleRunResults() if track_results else None
        previous_steps = (counter if counter > -1 else 0) * counter_len
        first_batch_size = None
        epoch_timer = time.time()

        for batch, data in data_iter:
            if 0 < max_steps and batch >= max_steps:
                break

            step_timer = time.time()
            batch_size = self._run_funcs.batch_size(data)  # type: int

            if first_batch_size is None:
                first_batch_size = batch_size

            should_log = (
                self._loggers
                and self._log_steps
                and self._log_steps > 0
                and batch % self._log_steps == 0
            )
            log_step = previous_steps + batch
            batch_results = self._runner_batch(
                counter, batch, batch_size, data, should_log, log_step, counter_len
            )

            if should_log:
                for loss, val in batch_results.items():
                    self._log_scalar(
                        "{}/{}".format(self._log_name, loss),
                        val.item(),
                        log_step,
                    )

                self._log_scalar(
                    "{}/Epoch Counter".format(self._log_name),
                    counter,
                    log_step,
                )
                self._log_scalar(
                    "{}/Batch Size".format(self._log_name),
                    batch_size,
                    log_step,
                )
                step_time = time.time() - step_timer
                self._log_scalar(
                    "{}/Seconds per step".format(self._log_name),
                    step_time,
                    log_step,
                )
                self._log_scalar(
                    "{}/Steps per second".format(self._log_name),
                    1.0 / step_time,
                    log_step,
                )

                if progress_steps:
                    remaining_steps = progress_steps - batch - 1
                    self._log_scalar(
                        "{}/Est remaining minutes".format(self._log_name),
                        (step_time * remaining_steps) / 60,
                        log_step,
                    )

            if results is not None:
                results.append(batch_results, batch_size)

        should_log = self._loggers and self._log_summary and results
        log_step = counter  # log under the counter step for the summaries

        if should_log:
            for loss in results.results.keys():
                val = results.result_mean(loss)
                self._log_scalar(
                    "{}/{} Summary".format(self._log_name, loss),
                    val.item(),
                    log_step,
                )

            self._log_scalar(
                "{}/Batch Size Summary".format(self._log_name),
                first_batch_size,
                log_step,
            )
            self._log_scalar(
                "{}/Minutes per epoch".format(self._log_name),
                (time.time() - epoch_timer) / 60,
                log_step,
            )

        self._runner_complete(results, should_log, log_step)

        return results

    def run_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        max_epochs: int,
        show_progress: bool = True,
        track_results: bool = True,
        max_steps: int = -1,
        gradient_accum_steps: int = 1,
    ):
        """
        Convenience function for evaluation over all the data in the given data loader
        for a specific epoch and making the progress visible.

        :param data_loader: the data loader used to gather batches to be run
            through the model
        :param epoch: the current evaluation epoch number
        :param show_progress: True to show a progress bar, False otherwise
        :param track_results: True to track and return the results of the training,
            False to return None
        :param max_steps: maximum number of steps/batches to run through,
            will stop after reaching this. if <= 0 then no restriction is placed
        :param gradient_accum_steps: Number of gradient accumulation steps to run before
            updating weights
        :return: the results of evaluation if track_results else None
        """
        return self.run(
            data_loader,
            "{} epoch {}/{}".format(self._log_name, epoch, max_epochs),
            epoch,
            show_progress,
            track_results,
            max_steps,
        )

    def _log_scalar(self, key: str, item: Any, step: int):
        for logger in self._loggers:
            logger.log_scalar(key, item, step)

    @abstractmethod
    def _runner_setup(self):
        raise NotImplementedError()

    @abstractmethod
    def _runner_batch(
        self,
        counter: int,
        batch: int,
        batch_size: int,
        data: Any,
        should_log: bool,
        log_step: int,
        counter_len: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def _runner_complete(
        self, results: ModuleRunResults, should_log: bool, log_step: int
    ):
        raise NotImplementedError()


class ModuleTrainer(ModuleRunner):
    """
    Container for running a module through training over a given data loader
    for specific settings.

    | Lifecycle:
    |   - data batch size callback
    |   - data to device callback
    |   - batch start hook
    |   - data model forward callback
    |   - batch forward hook
    |   - loss calculation
    |   - batch loss hook
    |   - model backward callback
    |   - batch backward hook
    |   - optimizer / gradient update
    |   - batch end hook

    :param module: the model to run training for
    :param device: the default device to run training on (where data will be copied to)
    :param loss: the loss functions callable used to calculate loss values
        after executing a forward pass
    :param optimizer: the optimizer used to apply gradient updates with
    :param num_accumulated_batches: number of batches to accumulate before
        updating the optimizer
    :param optim_closure: a closure passed into the optimizer on step
    :param loggers: list of loggers to log training results to
    :param log_name: the key to store all log files under
    :param log_steps: The number of steps (batches) to log at,
        ex 100 will log every 100 batches
    :param log_summary: True to log the final summary results after the run completes
    :param device_context: ModuleDeviceContext with settings to enable mixed precision
        using torch.cuda.amp or adjust losses when using DistributedDataParallel.
        Default settings do not use mixed precision or account for DDP.
        Will raise an exception if torch version does not support amp.
    """

    def __init__(
        self,
        module: Module,
        device: str,
        loss: Union[LossWrapper, Callable[[Any, Any], Tensor]],
        optimizer: Optimizer,
        num_accumulated_batches: int = 1,
        optim_closure: Union[None, Callable] = None,
        loggers: Optional[List[BaseLogger]] = None,
        log_name: str = "Train",
        log_steps: int = 100,
        log_summary: bool = True,
        device_context: ModuleDeviceContext = ModuleDeviceContext.default_context(),
    ):
        super().__init__(
            module,
            device,
            loss,
            loggers,
            log_name,
            log_steps,
            log_summary,
            device_context,
        )
        self._optimizer = optimizer
        self._num_accumulated_batches = num_accumulated_batches
        self._optim_closure = optim_closure

        self._accumulated = None

        if self.device_context.use_mixed_precision:
            if autocast is None or GradScaler is None:
                raise type(amp_import_error)(
                    amp_import_error.msg
                    + " autocast and GradScaler introduced in torch version 1.6.0."
                )
            if optim_closure is not None:
                raise RuntimeError(
                    "Optimizer closures are not currently supported when training "
                    "using torch.cuda.amp.GradScaler."
                )
            self._scaler = GradScaler()
        else:
            self._scaler = None

    @property
    def optimizer(self) -> Optimizer:
        """
        :return: the optimizer used to apply gradient updates with
        """
        return self._optimizer

    @property
    def num_accumulated_batches(self) -> int:
        """
        :return: number of batches to accumulate before updating the optimizer
        """
        return self._num_accumulated_batches

    @property
    def optim_closure(self) -> Union[None, Callable]:
        """
        :return: a closure passed into the optimizer on step
        """
        return self._optim_closure

    def _runner_setup(self):
        self._module = self._module.train()
        self._accumulated = 0

    def _runner_batch(
        self,
        counter: int,
        batch: int,
        batch_size: int,
        data: Any,
        should_log: bool,
        log_step: int,
        counter_len: int,
    ):
        if self._accumulated == 0:
            self._optimizer.zero_grad()

        # setup
        self._accumulated += 1
        data = self._run_funcs.to_device(data, self._device)
        self._run_hooks.invoke_batch_start(counter, batch, batch_size, data)

        forward_context = (
            autocast if self.device_context.use_mixed_precision else ExitStack
        )
        with forward_context():
            # forward steps
            pred = self._run_funcs.model_forward(data, self._module)
            self._run_hooks.invoke_batch_forward(counter, batch, batch_size, data, pred)

            # loss calculation
            losses = self._loss(data, pred)
            losses[DEFAULT_LOSS_KEY] /= self._num_accumulated_batches

            self._run_hooks.invoke_batch_loss(
                counter, batch, batch_size, data, pred, losses
            )

        # backward steps
        self._run_funcs.model_backward(losses, self._module, scaler=self._scaler)
        self._run_hooks.invoke_batch_backward(
            counter, batch, batch_size, data, pred, losses
        )

        losses[DEFAULT_LOSS_KEY] *= self._num_accumulated_batches

        # optimizer / gradients update
        if (
            self._accumulated == self._num_accumulated_batches
            or self._accumulated == counter_len
        ):
            if self.device_context.use_mixed_precision:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step(closure=self._optim_closure)
            self._accumulated = 0

        self._run_hooks.invoke_batch_end(counter, batch, batch_size, data, pred, losses)

        if should_log:
            self._log_scalar(
                "{}/Learning Rate".format(self._log_name),
                get_optim_learning_rate(self._optimizer),
                log_step,
            )

        return losses

    def _runner_complete(
        self, results: ModuleRunResults, should_log: bool, log_step: int
    ):
        if should_log:
            self._log_scalar(
                "{}/Learning Rate Summary".format(self._log_name),
                get_optim_learning_rate(self._optimizer),
                log_step,
            )


class ModuleTester(ModuleRunner):
    """
    Container for running a module through evaluation over a given data loader
    for specific settings.

    | Lifecycle:
    |   - data batch size callback
    |   - data to device callback
    |   - batch start hook
    |   - data model forward callback
    |   - batch forward hook
    |   - loss calculation
    |   - batch loss hook
    |   - batch end hook

    :param module: the model to run evaluation for
    :param device: the default device to run evaluation on
        (where data will be copied to)
    :param loss: the loss functions callable used to calculate loss values after
        executing a forward pass
    :param loggers: list of loggers to log training results to
    :param log_name: the key to store all log files under
    :param log_steps: The number of steps (batches) to log at,
        ex 100 will log every 100 batches
    :param log_summary: True to log the final summary results after the run completes
    :param device_context: ModuleDeviceContext with settings to enable mixed precision
        using torch.cuda.amp or adjust losses when using DistributedDataParallel.
        Default settings do not use mixed precision or account for DDP.
        Will raise an exception if torch version does not support amp.
    """

    def __init__(
        self,
        module: Module,
        device: str,
        loss: Union[LossWrapper, Callable[[Any, Any], Tensor]],
        loggers: Optional[List[BaseLogger]] = None,
        log_name: str = "Test",
        log_steps: int = 100,
        log_summary: bool = True,
        device_context: ModuleDeviceContext = ModuleDeviceContext.default_context(),
    ):
        super().__init__(
            module,
            device,
            loss,
            loggers,
            log_name,
            log_steps,
            log_summary,
            device_context,
        )

        if self.device_context.use_mixed_precision:
            if autocast is None or GradScaler is None:
                raise type(amp_import_error)(
                    amp_import_error.msg
                    + " autocast and GradScaler introduced in torch version 1.6.0."
                )

    def _runner_setup(self):
        self._module = self._module.eval()

    def _runner_batch(
        self,
        counter: int,
        batch: int,
        batch_size: int,
        data: Any,
        should_log: bool,
        log_step: int,
        counter_len: int,
    ):
        with torch.no_grad():
            # setup
            data = self._run_funcs.to_device(data, self._device)
            self._run_hooks.invoke_batch_start(counter, batch, batch_size, data)

            forward_context = (
                autocast if self.device_context.use_mixed_precision else ExitStack
            )
            with forward_context():
                # forward steps
                pred = self._run_funcs.model_forward(data, self._module)
                self._run_hooks.invoke_batch_forward(
                    counter, batch, batch_size, data, pred
                )

                # loss steps
                losses = self._loss(data, pred)

                self._run_hooks.invoke_batch_loss(
                    counter, batch, batch_size, data, pred, losses
                )

            self._run_hooks.invoke_batch_end(
                counter, batch, batch_size, data, pred, losses
            )

        return losses

    def _runner_complete(
        self, results: ModuleRunResults, should_log: bool, log_step: int
    ):
        pass
