import torch
from typing import Dict, Union
from torch import Tensor
from torch.nn import DataParallel, Module
from torch.optim.optimizer import Optimizer
from collections import OrderedDict


__all__ = ['copy_state_dict_value', 'copy_state_dict_linear', 'copy_state_dict_conv', 'copy_state_dict_batch_norm',
           'load_model', 'save_model']


def copy_state_dict_value(target_key: str, source_key: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                          delete_from_source: bool = False):
    if source_key not in source:
        raise Exception('{} not found in source dict'.format(source_key))

    target[target_key] = source[source_key]

    if delete_from_source:
        del source[source_key]


def copy_state_dict_linear(target_name: str, source_name: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                           bias: bool = True, delete_from_source: bool = False):
    copy_state_dict_value('{}.weight'.format(target_name), '{}.weight'.format(source_name),
                          target, source, delete_from_source)

    if bias:
        copy_state_dict_value('{}.bias'.format(target_name), '{}.bias'.format(source_name),
                              target, source, delete_from_source)


def copy_state_dict_conv(target_name: str, source_name: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                         bias: bool = True, delete_from_source: bool = False):
    copy_state_dict_value('{}.weight'.format(target_name), '{}.weight'.format(source_name),
                          target, source, delete_from_source)

    if bias:
        copy_state_dict_value('{}.bias'.format(target_name), '{}.bias'.format(source_name),
                              target, source, delete_from_source)


def copy_state_dict_batch_norm(target_name: str, source_name: str, target: Dict[str, Tensor], source: Dict[str, Tensor],
                               delete_from_source: bool = False):
    copy_state_dict_value('{}.weight'.format(target_name), '{}.weight'.format(source_name),
                          target, source, delete_from_source)
    copy_state_dict_value('{}.bias'.format(target_name), '{}.bias'.format(source_name),
                          target, source, delete_from_source)
    copy_state_dict_value('{}.running_mean'.format(target_name), '{}.running_mean'.format(source_name),
                          target, source, delete_from_source)
    copy_state_dict_value('{}.running_var'.format(target_name), '{}.running_var'.format(source_name),
                          target, source, delete_from_source)

    if delete_from_source and '{}.num_batches_tracked'.format(source_name) in source:
        del source['{}.num_batches_tracked'.format(source_name)]


def load_model(path: str, model: Module, optimizer: Optimizer = None, strict: bool = True):
    model_dict = torch.load(path, map_location='cpu')
    first_key = [key for key in model_dict['state_dict']][0]

    if first_key.startswith('module.'):
        # convert without module because we don't have module in our naming
        tmp_dict = OrderedDict()

        for key, value in model_dict['state_dict'].items():
            key = key[7:]
            tmp_dict[key] = value

        model_dict['state_dict'] = tmp_dict

    model.load_state_dict(model_dict['state_dict'], strict)

    if optimizer:
        optimizer.load_state_dict(model_dict['optimizer'])


def save_model(path: str, model: Module, optimizer: Optimizer = None, epoch: Union[int, None] = None):
    if isinstance(model, DataParallel):
        model = model.module

    save_dict = {
        'state_dict': OrderedDict()
    }

    # make sure we have the model state_dict on cpu
    for key, state in model.state_dict().items():
        copy = torch.zeros(state.shape)
        copy.copy_(state)
        save_dict['state_dict'][key] = copy

    if optimizer:
        save_dict['optimizer'] = optimizer.state_dict()

    if epoch:
        save_dict['epoch'] = epoch

    torch.save(save_dict, path)