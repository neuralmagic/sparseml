from typing import Dict, Callable, Tuple
from collections import OrderedDict
from tqdm import tqdm
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from torch.optim.optimizer import Optimizer

from .loss_wrapper import LossWrapper


__all__ = ['ModuleTrainer', 'train_module']


class ModuleTrainer(object):
    def __init__(self, module: Module, device: str, loss: LossWrapper, optimizer: Optimizer):
        self._module = module
        self._device = device
        self._loss = loss
        self._optimizer = optimizer

        self._batch_start_hooks = OrderedDict()
        self._batch_pred_hooks = OrderedDict()
        self._batch_loss_hooks = OrderedDict()
        self._batch_end_hooks = OrderedDict()

    @property
    def module(self) -> Module:
        return self._module

    @property
    def device(self) -> str:
        return self._device

    @property
    def loss(self) -> LossWrapper:
        return self._loss

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def register_batch_start_hook(self, hook: Callable[[int, int, Tuple[Tensor, ...], Tensor], None]):
        """
        :param hook: the callback for the hook, inputs are expected to take in (epoch, step, x_feature, y_lab)
        :return: the handle to remove the hook at a later time
        """
        handle = RemovableHandle(self._batch_start_hooks)
        self._batch_start_hooks[handle.id] = hook

        return handle

    def register_batch_pred_hook(self, hook: Callable[[int, int, Tuple[Tensor, ...], Tensor, Tensor], None]):
        """
        :param hook: the callback for the hook, inputs are expected to take in (epoch, step, x_feature, y_lab, y_pred)
        :return: the handle to remove the hook at a later time
        """
        handle = RemovableHandle(self._batch_pred_hooks)
        self._batch_pred_hooks[handle.id] = hook

        return handle

    def register_batch_loss_hook(self, hook: Callable[[int, int, Tuple[Tensor, ...],
                                                       Tensor, Tensor, Dict[str, Tensor]], None]):
        """
        :param hook: the callback for the hook, inputs are expected to take in
                     (epoch, step, x_feature, y_lab, y_pred, losses))
        :return: the handle to remove the hook at a later time
        """
        handle = RemovableHandle(self._batch_loss_hooks)
        self._batch_loss_hooks[handle.id] = hook

        return handle

    def register_batch_end_hook(self, hook: Callable[[int, int, Tuple[Tensor, ...],
                                                      Tensor, Tensor, Dict[str, Tensor]], None]):
        """
        :param hook: the callback for the hook, inputs are expected to take in
                     (epoch, step, x_feature, y_lab, y_pred, losses))
        :return: the handle to remove the hook at a later time
        """
        handle = RemovableHandle(self._batch_end_hooks)
        self._batch_end_hooks[handle.id] = hook

        return handle

    def train_epoch(self, data_loader: DataLoader, epoch: int):
        self._module = self._module.train()
        print('training for epoch {}'.format(epoch))
        step_count = 0

        for batch, (*x_feature, y_lab) in tqdm(enumerate(data_loader),
                                               desc='training epoch {}'.format(epoch),
                                               total=len(data_loader)):
            # copy next batch to the device we are using
            y_lab = y_lab.to(self.device)
            x_feature = tuple([dat.to(self.device) for dat in x_feature])
            batch_size = y_lab.shape[0]

            for hook in self._batch_start_hooks.values():
                hook(epoch, step_count, x_feature, y_lab)

            self.optimizer.zero_grad()
            y_pred = self.module(*x_feature)

            for hook in self._batch_pred_hooks.values():
                hook(epoch, step_count, x_feature, y_lab, y_pred)

            losses = self.loss(x_feature, y_lab, y_pred)  # type: Dict[str, Tensor]

            for hook in self._batch_loss_hooks.values():
                hook(epoch, step_count, x_feature, y_lab, y_pred, losses)

            losses['loss'].backward()
            self.optimizer.step(closure=None)

            for hook in self._batch_end_hooks.values():
                hook(epoch, step_count, x_feature, y_lab, y_pred, losses)

            step_count += batch_size


def train_module(module: Module, device: str, loss: LossWrapper, optimizer: Optimizer,
                 data_loader: DataLoader, epoch: int):
    trainer = ModuleTrainer(module, device, loss, optimizer)
    trainer.train_epoch(data_loader, epoch)
