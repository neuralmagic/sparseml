from typing import List, Tuple
import copy
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader

from .loss_wrapper import LossWrapper


__all__ = ['lr_analysis']


def _endless_loader(data_loader: DataLoader):
    while True:
        for data in data_loader:
            yield data


def lr_analysis(module: Module, device: str, train_data: DataLoader, loss_wrapper: LossWrapper,
                batches_per_sample: int = 10, lr_mult: float = 1.1, init_lr: float = 1e-9, final_lr: float = 1e0,
                sgd_momentum: float = 0.9, sgd_dampening: float = 0.0, sgd_weight_decay: float = 1e-4,
                sgd_nesterov: bool = True) -> List[Tuple[float, Tensor]]:
    lr_module = copy.deepcopy(module.to('cpu'))
    lr_module = lr_module.to(device).train()
    optimizer = SGD(lr_module.parameters(), init_lr, sgd_momentum, sgd_dampening, sgd_weight_decay, sgd_nesterov)

    data_loader = _endless_loader(train_data)
    check_lrs = [init_lr]

    while check_lrs[-1] < final_lr:
        check_lrs.append(check_lrs[-1] * lr_mult)

    analysis = []

    for check_lr in tqdm(check_lrs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = check_lr

        losses = []

        for _ in range(batches_per_sample):
            (*x_feature, y_lab) = next(data_loader)
            y_lab = y_lab.to(device)
            x_feature = tuple([dat.to(device) for dat in x_feature])
            optimizer.zero_grad()
            y_pred = lr_module(*x_feature)
            loss = loss_wrapper(x_feature, y_lab, y_pred)
            loss['loss'].backward()
            optimizer.step(closure=None)

            loss = loss['loss'].detach_().cpu()
            loss = loss.repeat(y_lab.shape[0])
            losses.append(loss)

        analysis.append((check_lr, torch.stack(losses)))

    return analysis
