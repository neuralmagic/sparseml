from typing import Union, Tuple, Iterable, Dict, Any
import os
import errno

from torch import Tensor


__all__ = ['tensors_batch_size', 'tensors_to_device',
           'clean_path', 'create_dirs', 'create_parent_dirs']


def tensors_batch_size(tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]):
    if isinstance(tensors, Tensor):
        return tensors.shape[0]

    if isinstance(tensors, Dict):
        for key, tens in tensors.items():
            if isinstance(tens, Tensor):
                return tens.shape[0]

    if isinstance(tensors, Iterable):
        for tens in tensors:
            if isinstance(tens, Tensor):
                return tens.shape[0]

    return -1


def tensors_to_device(tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]],
                       device: str) -> Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]:
    if isinstance(tensors, Tensor):
        return tensors.to(device)

    if isinstance(tensors, Dict):
        return {key: tens.to(device) for key, tens in tensors.items()}

    if isinstance(tensors, Tuple):
        return tuple(tens.to(device) for tens in tensors)

    if isinstance(tensors, Iterable):
        return [tens.to(device) for tens in tensors]

    raise ValueError('unrecognized type for tensors given of {}'.format(tensors.__class__.__name__))


def clean_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def create_dirs(path: str):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            # Unexpected OSError, re-raise.
            raise


def create_parent_dirs(path: str):
    parent = os.path.dirname(path)
    create_dirs(parent)
