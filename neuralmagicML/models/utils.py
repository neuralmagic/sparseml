from typing import Dict
from torch import Tensor


__all__ = ['copy_state_dict_value', 'copy_state_dict_linear', 'copy_state_dict_conv', 'copy_state_dict_batch_norm']


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
