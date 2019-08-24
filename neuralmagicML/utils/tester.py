from typing import Dict, List, Callable, Tuple
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from .loss_wrapper import LossWrapper


__all__ = ['ModuleTestResults', 'ModuleTester', 'test_module']


class ModuleTestResults(object):
    def __init__(self):
        self._results = {}

    @property
    def results(self) -> Dict[str, List[Tensor]]:
        return self._results

    def result(self, key: str) -> List[Tensor]:
        return self._results[key]

    def result_mean(self, key: str) -> Tensor:
        res = self.result(key)

        return torch.mean(torch.cat(res))

    def result_std(self, key: str) -> Tensor:
        res = self.result(key)

        return torch.std(torch.cat(res))

    def append(self, losses: Dict[str, Tensor], batch_size: int):
        for key, val in losses.items():
            if key not in self._results:
                self._results[key] = []

            result = val.detach_().cpu()
            result = result.repeat(batch_size)
            self._results[key].append(result)


class ModuleTester(object):
    def __init__(self, module: Module, device: str, loss: LossWrapper):
        self._module = module
        self._device = device
        self._loss = loss

        self._batch_start_hooks = OrderedDict()
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

    def register_batch_start_hook(self, hook: Callable[[int, int, Tuple[Tensor, ...], Tensor], None]):
        """
        :param hook: the callback for the hook, inputs are expected to take in (epoch, step, x_feature, y_lab)
        :return: the handle to remove the hook at a later time
        """
        handle = RemovableHandle(self._batch_start_hooks)
        self._batch_start_hooks[handle.id] = hook

        return handle

    def register_batch_end_hook(self, hook: Callable[[int, int, Tuple[Tensor, ...],
                                                      Tensor, Tensor, Dict[str, Tensor]], None]):
        """
        :param hook: the callback for the hook, inputs are expected to take in
                     (epoch, step, x_feature, y_lab, y_pred, losses)
        :return: the handle to remove the hook at a later time
        """
        handle = RemovableHandle(self._batch_end_hooks)
        self._batch_end_hooks[handle.id] = hook

        return handle

    def test_epoch(self, data_loader: DataLoader, epoch: int) -> ModuleTestResults:
        self._module = self._module.eval()
        results = ModuleTestResults()
        print('testing for epoch {}'.format(epoch))
        step_count = 0

        with torch.no_grad():
            for batch, (*x_feature, y_lab) in tqdm(enumerate(data_loader)):
                y_lab = y_lab.to(self.device)
                x_feature = tuple([dat.to(self.device) for dat in x_feature])
                batch_size = y_lab.shape[0]

                for hook in self._batch_start_hooks.values():
                    hook(epoch, step_count, x_feature, y_lab)

                y_pred = self.module(*x_feature)
                losses = self.loss(x_feature, y_lab, y_pred)  # type: Dict[str, Tensor]
                results.append(losses, batch_size)

                for hook in self._batch_end_hooks.values():
                    hook(epoch, step_count, x_feature, y_lab, y_pred, losses)

                step_count += batch_size

        return results


def test_module(module: Module, device: str, loss: LossWrapper,
                data_loader: DataLoader, epoch: int) -> ModuleTestResults:
    tester = ModuleTester(module, device, loss)
    results = tester.test_epoch(data_loader, epoch)

    return results
