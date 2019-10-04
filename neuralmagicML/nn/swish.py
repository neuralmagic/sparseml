from torch import Tensor, sigmoid
from torch.nn import Module

__all__ = ['Swish']


def swish(x: Tensor):
    return x * sigmoid(x)


class Swish(Module):
    def forward(self, inp: Tensor):
        return swish(inp)
