"""
Code related to running a module through training and testing over a dataset
Allows reporting of progress and override functions and hooks
"""

from typing import Callable, Any, Dict, Union, List
from collections import OrderedDict
from tqdm import auto

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.hooks import RemovableHandle

from neuralmagicML.pytorch.utils.helpers import (
    tensors_to_device,
    tensors_batch_size,
    tensors_module_forward,
)
from neuralmagicML.pytorch.utils.logger import PyTorchLogger
from neuralmagicML.pytorch.utils.loss import DEFAULT_LOSS_KEY


__all__ = [
    "def_model_backward",
    "ModuleRunFuncs",
    "ModuleRunHooks",
    "ModuleRunResults",
    "ModuleTester",
    "ModuleTrainer",
]


def def_model_backward(losses: Dict[str, Tensor], model: Module):
    """
    Default function to perform a backwards pass for a model and the calculated losses
    Calls backwards for the DEFAULT_LOSS_KEY in losses Dict

    :param model: the model to run the backward for
    :param losses: the losses dictionary containing named tensors,
                   DEFAULT_LOSS_KEY is expected to exist and backwards is called on that
    """
    # assume loss is at default loss key
    losses[DEFAULT_LOSS_KEY].backward()


class ModuleRunHooks(object):
    """
    Container for hooks that can be added to module runs like training and testing for different stages
    of running a batch through a model

    Lifecycle:
        - data batch size callback
        - data to device callback
        - batch start hook
        - data model forward callback
        - batch forward hook
        - loss calculation
        - batch loss hook
        - model backward callback
        - batch backward hook
        - optimizer / gradient update
        - batch end hook
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
        where counter is passed in to the run (ex: epoch), step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch, data is the data output from the loader

        :param hook: the hook to add that is called into when reached in the batch process
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
        where counter is passed in to the run (ex: epoch), step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch, data is the data output from the loader,
        pred is the result from the model after the forward

        :param hook: the hook to add that is called into when reached in the batch process
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
        where counter is passed in to the run (ex: epoch), step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch, data is the data output from the loader,
        pred is the result from the model after the forward, losses are the resulting loss dictionary

        :param hook: the hook to add that is called into when reached in the batch process
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
        where counter is passed in to the run (ex: epoch), step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch, data is the data output from the loader,
        pred is the result from the model after the forward, losses are the resulting loss dictionary

        :param hook: the hook to add that is called into when reached in the batch process
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
        where counter is passed in to the run (ex: epoch), step_count is the number of items run so far,
        batch_size is the number of elements fed in the batch, data is the data output from the loader,
        pred is the result from the model after the forward, losses are the resulting loss dictionary

        :param hook: the hook to add that is called into when reached in the batch process
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
    Functions used as callables to calculate or perform necessary operations for running a model
    through training or testing.

     Lifecycle:
        - data batch size callback
        - data to device callback
        - batch start hook
        - data model forward callback
        - batch forward hook
        - loss calculation
        - batch loss hook
        - model backward callback
        - batch backward hook
        - optimizer / gradient update
        - batch end hook
    """

    def __init__(self):
        self._batch_size = tensors_batch_size
        self._to_device = tensors_to_device
        self._model_forward = tensors_module_forward
        self._model_backward = def_model_backward

    @property
    def batch_size(self) -> Callable[[Any], int]:
        """
        Used to calculate the batch size of a given grouping of tensors
        Expected to be called with the output from a data loader and then return an int representing the batch size
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: Callable[[Any], int]):
        """
        Used to calculate the batch size of a given grouping of tensors
        Expected to be called with the output from a data loader
         then return an int representing the batch size

        :param value: the callable used to calculate batch size for a grouping of tensors
        """
        self._batch_size = value

    @property
    def to_device(self) -> Callable[[Any, str], Any]:
        """
        Used to place a given grouping of tensors onto the proper device
        Expected to be called with the output from a data loader and the desired device as a string
         then return the grouping on the proper device
        """
        return self._to_device

    @to_device.setter
    def to_device(self, value: Callable[[Any, str], Any]):
        """
        Used to place a given grouping of tensors onto the proper device
        Expected to be called with the output from a data loader and the desired device as a string
         then return the grouping on the proper device

        :param value: the callable used to place a grouping of tensors onto the proper device
        """
        self._to_device = value

    @property
    def model_forward(self) -> Callable[[Any, Module], Any]:
        """
        Used to propagate a given grouping of tensors through a model and return the result
        Expected to be called with the model and the output from a data loader
         then return the result from the model forward pass
        """
        return self._model_forward

    @model_forward.setter
    def model_forward(self, value: Callable[[Any, Module], Any]):
        """
        Used to propagate a given grouping of tensors through a model and return the result
        Expected to be called with the model and the output from a data loader
         then return the result from the model forward pass

        :param value: the callable used to run a grouping of tensors through a model
        :return: the result of running the data through the model
        """
        self._model_forward = value

    @property
    def model_backward(self) -> Callable[[Dict[str, Tensor], Module], None]:
        """
        Used to call backward for a given model and the calculated losses
        Expected to be called with the model and the output from the loss function as a dict mapping of names to tensors
         returns nothing
        """
        return self._model_backward

    @model_backward.setter
    def model_backward(self, value: Callable[[Dict[str, Tensor], Module], None]):
        """
        Used to call backward for a given model and the calculated losses
        Expected to be called with the model and the output from the loss function as a dict mapping of names to tensors
         returns nothing

        :param value: the callable used to run a backwards pass for the given loss functions
        """
        self._model_backward = value

    def copy(self, run_funcs):
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

        :return: a dictionary containing a mapping of name (str) to a list of tensors that were recorded for that loss
        """
        return self._results

    def result(self, key: str) -> List[Tensor]:
        """
        The result of a single loss function

        :param key: the name of the loss function to get the results for
        :return: a list of tensors containing all of the results for that loss
        """
        return self._results[key]

    def result_mean(self, key: str) -> Tensor:
        """
        The mean result of a single loss function

        :param key: the name of the loss function to get the mean result for
        :return: a single tensor containing the average of all the results for that loss
        """
        res = self.result(key)

        return torch.mean(torch.cat(res))

    def result_std(self, key: str) -> Tensor:
        """
        The standard deviation of the result for a single loss function

        :param key: the name of the loss function to get the standard deviation result for
        :return: a single tensor containing the standard deviation of all the results for that loss
        """
        res = self.result(key)

        return torch.std(torch.cat(res))

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


class ModuleTrainer(object):
    """
    Container for running a module through training over a given data loader for specific settings

    Lifecycle:
        - data batch size callback
        - data to device callback
        - batch start hook
        - data model forward callback
        - batch forward hook
        - loss calculation
        - batch loss hook
        - model backward callback
        - batch backward hook
        - optimizer / gradient update
        - batch end hook
    """

    def __init__(
        self,
        module: Module,
        device: str,
        loss: Callable[[Any, Any], Dict[str, Tensor]],
        optimizer: Optimizer,
        num_accumulated_batches: int = 1,
        optim_closure: Union[None, Callable] = None,
        loggers: List[PyTorchLogger] = None,
        log_name: str = "Train/",
        log_batches: bool = True,
    ):
        """
        :param module: the model to run training for
        :param device: the default device to run training on (where data will be copied to)
        :param loss: the loss functions callable used to calculate loss values after executing a forward pass
        :param optimizer: the optimizer used to apply gradient updates with
        :param num_accumulated_batches: number of batches to accumulate before updating the optimizer
        :param optim_closure: a closure passed into the optimizer on step
        :param loggers: list of loggers to log training results to
        :param log_name: the key to store all log files under
        :param log_batches: True to log each batch step,
                            False to only log the end result (track_results must be True then)
        """
        self._module = module
        self._device = device
        self._loss = loss
        self._optimizer = optimizer
        self._num_accumulated_batches = num_accumulated_batches
        self._optim_closure = optim_closure
        self._loggers = loggers
        self._log_name = log_name
        self._log_batches = log_batches

        self._run_funcs = ModuleRunFuncs()
        self._run_hooks = ModuleRunHooks()

    @property
    def module(self) -> Module:
        """
        :return: the model to run training for
        """
        return self._module

    @property
    def device(self) -> str:
        """
        :return: the default device to run training on (where data will be copied to)
        """
        return self._device

    @property
    def loss(self) -> Callable[[Any, Any], Dict[str, Tensor]]:
        """
        :return: the loss functions callable used to calculate loss values after executing a forward pass
        """
        return self._loss

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

    @property
    def run_funcs(self) -> ModuleRunFuncs:
        """
        :return: functions used while running training of the model as callbacks to do certain stages
        """
        return self._run_funcs

    @property
    def run_hooks(self) -> ModuleRunHooks:
        """
        :return: hooks used while running traiiniing of the model to receive intermediate results
        """
        return self._run_hooks

    def run(
        self,
        data_loader: DataLoader,
        desc: str,
        counter: int = -1,
        show_progress: bool = False,
        track_results: bool = False,
    ) -> Union[None, ModuleRunResults]:
        """
        Run training over all the data in the given data loader

        :param data_loader: the data loader used to gather batches to be run through the model
        :param desc: description used in the progress indicator
        :param counter: counter passed to the hooks for external state keeping (ex: epoch)
        :param show_progress: True to show a progress bar, False otherwise
        :param track_results: True to track and return the results of the training, False to return None
        :return: the results of training if track_results else None
        """
        self._module = self._module.train()
        step_count = 0
        step_size = -1
        accumulated = 0
        data_iter = (
            enumerate(data_loader)
            if not show_progress
            else auto.tqdm(enumerate(data_loader), desc=desc, total=len(data_loader))
        )
        results = ModuleRunResults() if track_results else None

        for batch, data in data_iter:
            # setup
            accumulated += 1
            batch_size = self._run_funcs.batch_size(data)  # type: int
            step_size = max(batch_size, step_size)
            data = self._run_funcs.to_device(data, self._device)
            self._run_hooks.invoke_batch_start(counter, step_count, batch_size, data)

            # optimizer / gradients reset
            if accumulated == self._num_accumulated_batches:
                self._optimizer.zero_grad()

            # forward steps
            pred = self._run_funcs.model_forward(data, self._module)
            self._run_hooks.invoke_batch_forward(
                counter, step_count, batch_size, data, pred
            )

            # backward steps
            losses = self._loss(data, pred)
            self._run_hooks.invoke_batch_loss(
                counter, step_count, batch_size, data, pred, losses
            )

            self._run_funcs.model_backward(losses, self._module)
            self._run_hooks.invoke_batch_backward(
                counter, step_count, batch_size, data, pred, losses
            )

            # optimizer / gradients update
            if accumulated == self._num_accumulated_batches:
                self._optimizer.step(closure=self._optim_closure)
                accumulated = 0

            self._run_hooks.invoke_batch_end(
                counter, step_count, batch_size, data, pred, losses
            )
            step_count += batch_size

            if self._log_batches and self._loggers:
                previous_steps = (
                    (counter if counter > -1 else 0) * len(data_loader) * step_size
                )
                log_step = previous_steps + step_count
                for loss, val in losses.items():
                    for logger in self._loggers:
                        logger.log_scalar(
                            "{}{}".format(self._log_name, loss), val.item(), log_step
                        )

            if results is not None:
                results.append(losses, batch_size)

        if results is not None and not self._log_batches and self._loggers:
            log_step = counter
            for loss in results.results.keys():
                val = results.result_mean(loss)
                for logger in self._loggers:
                    logger.log_scalar(
                        "{}{}".format(self._log_name, loss), val.item(), log_step
                    )

        return results

    def run_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        show_progress: bool = True,
        track_results: bool = False,
    ):
        """
        Convenience function for training over all the data in the given data loader for a specific epoch
         and making the progress visible

        :param data_loader: the data loader used to gather batches to be run through the model
        :param epoch: the current training epoch number
        :param show_progress: True to show a progress bar, False otherwise
        :param track_results: True to track and return the results of the training, False to return None
        :return: the results of training if track_results else None
        """
        return self.run(
            data_loader,
            "training epoch {}".format(epoch),
            epoch,
            show_progress,
            track_results,
        )


class ModuleTester(object):
    """
    Container for running a module through evaluation over a given data loader for specific settings

    Lifecycle:
        - data batch size callback
        - data to device callback
        - batch start hook
        - data model forward callback
        - batch forward hook
        - loss calculation
        - batch loss hook
        - batch end hook
    """

    def __init__(
        self,
        module: Module,
        device: str,
        loss: Callable[[Any, Any], Dict[str, Tensor]],
        loggers: List[PyTorchLogger] = None,
        log_name: str = "Test/",
        log_batches: bool = False,
    ):
        """
        :param module: the model to run evaluation for
        :param device: the default device to run evaluation on (where data will be copied to)
        :param loss: the loss functions callable used to calculate loss values after executing a forward pass
        :param loggers: list of loggers to log training results to
        :param log_name: the key to store all log files under
        :param log_batches: True to log each batch step,
                            False to only log the end result (track_results must be True then)
        """
        self._module = module
        self._device = device
        self._loss = loss
        self._loggers = loggers
        self._log_name = log_name
        self._log_batches = log_batches

        self._run_funcs = ModuleRunFuncs()
        self._run_hooks = ModuleRunHooks()

    @property
    def module(self) -> Module:
        """
        :return: the model to run evaluation for
        """
        return self._module

    @property
    def device(self) -> str:
        """
        :return: the default device to run evaluation on (where data will be copied to)
        """
        return self._device

    @property
    def loss(self) -> Callable[[Any, Any], Dict[str, Tensor]]:
        """
        :return: the loss functions callable used to calculate loss values after executing a forward pass
        """
        return self._loss

    @property
    def run_funcs(self) -> ModuleRunFuncs:
        """
        :return: functions used while running evaluation of the model as callbacks to do certain stages
        """
        return self._run_funcs

    @property
    def run_hooks(self) -> ModuleRunHooks:
        """
        :return: hooks used while running evaluation of the model to receive intermediate results
        """
        return self._run_hooks

    def run(
        self,
        data_loader: DataLoader,
        desc: str,
        counter: int = -1,
        show_progress: bool = False,
        track_results: bool = True,
    ) -> Union[None, ModuleRunResults]:
        """
        Run evaluation over all the data in the given data loader

        :param data_loader: the data loader used to gather batches to be run through the model
        :param desc: description used in the progress indicator
        :param counter: counter passed to the hooks for external state keeping (ex: epoch)
        :param show_progress: True to show a progress bar, False otherwise
        :param track_results: True to track and return the results of the evaluation, False to return None
        :return: the results of evaluation if track_results else None
        """
        self._module = self._module.eval()
        step_count = 0
        data_iter = (
            enumerate(data_loader)
            if not show_progress
            else auto.tqdm(enumerate(data_loader), desc=desc, total=len(data_loader))
        )
        results = ModuleRunResults() if track_results else None

        with torch.no_grad():
            for batch, data in data_iter:
                # setup
                batch_size = self._run_funcs.batch_size(data)  # type: int
                data = self._run_funcs.to_device(data, self._device)
                self._run_hooks.invoke_batch_start(
                    counter, step_count, batch_size, data
                )

                # forward steps
                pred = self._run_funcs.model_forward(data, self._module)
                self._run_hooks.invoke_batch_forward(
                    counter, step_count, batch_size, data, pred
                )

                # backward steps
                losses = self._loss(data, pred)
                self._run_hooks.invoke_batch_loss(
                    counter, step_count, batch_size, data, pred, losses
                )

                self._run_hooks.invoke_batch_end(
                    counter, step_count, batch_size, data, pred, losses
                )
                step_count += batch_size

                if self._log_batches and self._loggers:
                    previous_steps = (
                        (counter if counter > -1 else 0) * len(data_loader) * batch_size
                    )
                    log_step = previous_steps + step_count
                    for loss, val in losses.items():
                        for logger in self._loggers:
                            logger.log_scalar(
                                "{}{}".format(self._log_name, loss),
                                val.item(),
                                log_step,
                            )

                if results is not None:
                    results.append(losses, batch_size)

        if results is not None and not self._log_batches and self._loggers:
            log_step = counter
            for loss in results.results.keys():
                val = results.result_mean(loss)
                for logger in self._loggers:
                    logger.log_scalar(
                        "{}{}".format(self._log_name, loss), val.item(), log_step
                    )

        return results

    def run_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        show_progress: bool = True,
        track_results: bool = True,
    ):
        """
        Convenience function for evaluation over all the data in the given data loader for a specific epoch
         and making the progress visible

        :param data_loader: the data loader used to gather batches to be run through the model
        :param epoch: the current evaluation epoch number
        :param show_progress: True to show a progress bar, False otherwise
        :param track_results: True to track and return the results of the training, False to return None
        :return: the results of evaluation if track_results else None
        """
        return self.run(
            data_loader,
            "testing epoch {}".format(epoch),
            epoch,
            show_progress,
            track_results,
        )
