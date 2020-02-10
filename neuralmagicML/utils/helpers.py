from typing import Union, Tuple, Iterable, Dict, Any, List
import os
import sys
import errno
import collections
import random
import numpy

import torch
from torch import Tensor
from torch.nn import Module, Linear
from torch.nn.modules.conv import _ConvNd


__all__ = [
    'ALL_TOKEN',
    'flatten_list', 'convert_to_bool', 'validate_str_list',
    'INTERPOLATION_FUNCS', 'interpolate',
    'clean_path', 'create_dirs', 'create_parent_dirs',
    'tensors_batch_size', 'tensors_to_device',
    'threshold_for_sparsity', 'mask_from_threshold', 'mask_from_sparsity', 'mask_from_tensor',
    'tensor_density', 'tensor_sparsity', 'tensor_sample',
    'get_layer', 'get_terminal_layers', 'get_conv_layers', 'get_linear_layers'
]


ALL_TOKEN = "__ALL__"


##############################
#
# general python helper functions
#
##############################


def flatten_list(li):
    def _flatten_gen(_li):
        for el in _li:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from _flatten_gen(el)
            else:
                yield el

    return list(_flatten_gen(li))


def convert_to_bool(val: Any):
    return bool(val) if not isinstance(val, str) else bool(val) and 'f' not in val.lower()


def validate_str_list(val: Union[str, List[str]], val_name: str, par_name: str) -> Union[str, List[str]]:
    if isinstance(val, List):
        return flatten_list(val)

    if isinstance(val, str):
        if val.upper() != '__ALL__':
            raise ValueError('unsupported {} string given in {}: {}'.format(val_name, par_name, val))

        return val.upper()

    raise ValueError('unsupported type given for {} in {}: {}'.format(val_name, par_name, val))


INTERPOLATION_FUNCS = ['linear', 'cubic', 'inverse_cubic', 'geometric']


def interpolate(x_cur: float, x0: float, x1: float, y0: Any, y1: Any, inter_func: str = 'linear') -> Any:
    if inter_func not in INTERPOLATION_FUNCS:
        raise ValueError('unsupported inter_func given of {} must be one of {}'
                         .format(inter_func, INTERPOLATION_FUNCS))

    # convert our x to 0-1 range since equations are designed to fit in (0,0)-(1,1) space
    x_per = (x_cur - x0) / (x1 - x0)

    # map x to y using the desired function in (0,0)-(1,1) space
    if inter_func == 'linear':
        y_per = x_per
    elif inter_func == 'cubic':
        # https://www.wolframalpha.com/input/?i=1-(1-x)%5E3+from+0+to+1
        y_per = 1 - (1 - x_per) ** 3
    elif inter_func == 'inverse_cubic':
        # https://www.wolframalpha.com/input/?i=1-(1-x)%5E(1%2F3)+from+0+to+1
        y_per = 1 - (1 - x_per) ** (1/3)
    else:
        raise ValueError('unsupported inter_func given of {} in interpolate'.format(inter_func))

    if y_per <= 0.0 + sys.float_info.epsilon:
        return y0

    if y_per >= 1.0 - sys.float_info.epsilon:
        return y1

    # scale the threshold based on what we want the current to be
    return y_per * (y1 - y0) + y0


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


##############################
#
# pytorch tensor helper functions
#
##############################


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


def threshold_for_sparsity(tens: Tensor, sparsity: float) -> Tensor:
    bottomk, _ = torch.topk(tens.abs().view(-1), int(sparsity * tens.numel()), largest=False, sorted=True)
    thresh = bottomk[-1]

    return thresh


def mask_from_threshold(tens: Tensor, threshold: Union[float, Tensor]) -> Tensor:
    return torch.gt(torch.abs(tens), threshold).type(tens.type())


def mask_from_sparsity(tens: Tensor, sparsity: float) -> Tensor:
    threshold = threshold_for_sparsity(tens, sparsity)

    if threshold.item() > 0.0:
        return mask_from_threshold(tens, threshold)

    # too many zeros so will go over the already given sparsity and choose which zeros to not keep in mask at random
    zero_indices = (tens == 0.0).nonzero()
    rand_indices = list(range(zero_indices.shape[0]))
    random.shuffle(rand_indices)
    num_elem = tens.numel()
    num_mask = int(num_elem * sparsity)
    rand_indices = rand_indices[:num_mask]
    rand_indices = tens.new_tensor(rand_indices, dtype=torch.int64)
    zero_indices = zero_indices[rand_indices, :]
    mask = tens.new_ones(tens.shape).type(tens.type())
    mask[zero_indices.split(1, dim=1)] = 0

    return mask


def mask_from_tensor(tens: Tensor) -> Tensor:
    return torch.ne(tens, 0.0).type(tens.type())


def tensor_density(tens: Tensor, dim: Union[None, int, List[int], Tuple[int, ...]] = None) -> Tensor:
    density = (tensor_sparsity(tens, dim) - 1.0) * -1.0

    return density


def tensor_sparsity(tens: Tensor, dim: Union[None, int, List[int], Tuple[int, ...]] = None) -> Tensor:
    if dim is None:
        zeros = (tens == 0).sum()
        total = tens.numel()
    else:
        if isinstance(dim, int):
            dim = [dim]

        if max(dim) >= len(tens.shape):
            raise ValueError('Unsupported dim given of {} in {} for tensor shape {}'.format(max(dim), dim, tens.shape))

        sum_dims = [ind for ind in range(len(tens.shape)) if ind not in dim]
        zeros = (tens == 0).sum(dim=sum_dims)
        total = numpy.prod([tens.shape[ind] for ind in range(len(tens.shape)) if ind not in dim])

        if dim != [ind for ind in range(len(dim))]:
            # put the desired dimension(s) at the front
            zeros = zeros.permute(*dim, *[ind for ind in range(len(zeros.shape)) if ind not in dim]).contiguous()

    return zeros.float() / float(total)


def tensor_sample(tens: Tensor, sample_size: int, dim: Union[None, int, List[int], Tuple[int, ...]] = None) -> Tensor:
    if sample_size < 1:
        raise ValueError('improper sample size given of {}'.format(sample_size))

    if dim is None:
        indices = tens.new_zeros((sample_size,)).long().random_(0, tens.numel())
        samples = tens.view(-1)[indices]

        return samples

    if isinstance(dim, int):
        dim = [dim]

    if max(dim) >= len(tens.shape):
        raise ValueError('Unsupported dim given of {} in {} for tensor shape {}'.format(max(dim), dim, tens.shape))

    if dim != [ind for ind in range(len(dim))]:
        # put the desired dimension(s) at the front to sample from
        tens = tens.permute(*dim, *[ind for ind in range(len(tens.shape)) if ind not in dim]).contiguous()
        dim = [ind for ind in range(len(dim))]
    elif not tens.is_contiguous():
        tens = tens.contiguous()

    num_indices = int(numpy.prod([tens.shape[ind] for ind in range(len(dim))]))
    elem_per_ind = int(numpy.prod([tens.shape[ind] for ind in range(len(dim), len(tens.shape))]))
    # create a new tensor with offsets set for each of our elements that we are indexing
    indices = tens.new_tensor([ind * elem_per_ind for ind in range(num_indices)], dtype=torch.long).unsqueeze(1)
    # now broadcast it across to the total number of elements we should end with
    indices = indices * tens.new_ones((num_indices, sample_size), dtype=torch.long)
    # finally add in a random number within the available range per index
    indices += tens.new_zeros((num_indices, sample_size), dtype=torch.long).random_(0, elem_per_ind)
    # get our samples
    samples = tens.view(-1)[indices.view(-1)]
    # reshape for the proper dimension
    samples = samples.view(*(tens.shape[ind] for ind in dim), sample_size)

    return samples


##############################
#
# pytorch module helper functions
#
##############################


def get_layer(name: str, module: Module) -> Module:
    layers = name.split('.')
    layer = module

    for name in layers:
        layer = layer.__getattr__(name)

    return layer


def get_terminal_layers(module: Module) -> List[Module]:
    terminal = []

    for mod_name, mod in module.named_modules():
        # check if it is a root node (only has itself in named_modules)
        child_count = 0
        for _, __ in mod.named_modules():
            child_count += 1

        if child_count != 1:
            continue

        terminal.append(mod)

    return terminal


def get_conv_layers(module: Module) -> Dict[str, Module]:
    convs = {}

    for name, mod in module.named_modules():
        if isinstance(mod, _ConvNd):
            convs[name] = mod

    return convs


def get_linear_layers(module: Module) -> Dict[str, Module]:
    linears = {}

    for name, mod in module.named_modules():
        if isinstance(mod, Linear):
            linears[name] = mod

    return linears
