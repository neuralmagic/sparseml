from collections import OrderedDict

from torch import Tensor
from torch.nn import (
    Module, Sequential, Conv2d, AdaptiveAvgPool2d, Sigmoid, ReLU)

__all__ = ['SqueezeExcite']


class SqueezeExcite(Module):
    def __init__(self,
                 expanded_channels: int,
                 squeezed_channels: int):
        super().__init__()
        self.se = Sequential(OrderedDict([
            ('squeeze', AdaptiveAvgPool2d(1)),
            ('reduce', Sequential(OrderedDict([
                ('conv', Conv2d(in_channels=expanded_channels, out_channels=squeezed_channels, kernel_size=1)),
                ('act', ReLU(inplace=True))
            ]))),
            ('expand', Sequential(OrderedDict([
                ('conv', Conv2d(in_channels=squeezed_channels, out_channels=expanded_channels, kernel_size=1)),
                ('act', Sigmoid())
            ])))
        ]))

    def forward(self, inp: Tensor):
        out = self.se(inp)

        return out
