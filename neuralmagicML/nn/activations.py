from typing import Union
from torch import Tensor
from torch.nn import Module, PReLU, LeakyReLU
from torch.nn import ReLU as TReLU
from torch.nn import ReLU6 as TReLU6
import torch.nn.functional as TF


__all__ = ['ReLU', 'ReLU6', 'Swish', 'swish',
           'create_activation', 'replace_activation', 'is_activation']


class ReLU(TReLU):
    def __init__(self, num_channels: int, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.num_channels = num_channels


class ReLU6(TReLU6):
    def __init__(self, num_channels: int, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.num_channels = num_channels


def swish(x: Tensor):
    return x * TF.sigmoid(x)


class Swish(Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, inp: Tensor):
        return swish(inp)


def replace_activation(module: Module, key: str, act_type: str,
                       inplace: bool = False, num_channels: Union[int, None] = None, **kwargs) -> Module:
    layer = module
    layers = key.split('.')

    for lay in layers[:-1]:
        layer = layer.__getattr__(lay)

    cur = layer.__getattr__(layers[-1])

    if num_channels is None and hasattr(cur, 'num_channels'):
        num_channels = cur.num_channels
    elif num_channels is None and hasattr(cur, 'num_parameters'):
        num_channels = cur.num_parameters

    act = create_activation(act_type, inplace=inplace, num_channels=num_channels, **kwargs)
    layer.__setattr__(layers[-1], act)

    return act


def create_activation(act_type: str, inplace: bool, num_channels: int, **kwargs) -> Module:
    act_type = act_type.lower()

    if act_type == 'relu':
        return ReLU(num_channels=num_channels, inplace=inplace)

    if act_type == 'relu6':
        return ReLU6(num_channels=num_channels, inplace=inplace)

    if act_type == 'prelu':
        return PReLU(num_parameters=num_channels, **kwargs)

    if act_type == 'lrelu':
        return LeakyReLU(inplace=inplace, **kwargs)

    if act_type == 'swish':
        return Swish(num_channels=num_channels)

    raise ValueError('unknown act_type given of {}'.format(act_type))


def is_activation(module: Module) -> bool:
    return isinstance(module, TReLU) or isinstance(module, TReLU6) \
            or isinstance(module, ReLU) or isinstance(module, ReLU6) \
            or isinstance(module, PReLU) or isinstance(module, LeakyReLU) \
            or isinstance(module, Swish)
