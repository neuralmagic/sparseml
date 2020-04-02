"""
Implementations for Squeeze Excite in PyTorch.
More information can be found in the paper
`here <https://arxiv.org/abs/1709.01507>`__.
"""

from collections import OrderedDict

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, AdaptiveAvgPool2d, Sigmoid

from neuralmagicML.pytorch.nn.activations import create_activation


__all__ = ["SqueezeExcite"]


class SqueezeExcite(Module):
    """
    Standard implementation for SqueezeExcite in PyTorch

    :param expanded_channels: the number of channels to expand to in the SE layer
    :param squeezed_channels: the number of channels to squeeze down to in the SE layer
    :param act_type: the activation type to use in the SE layer; options:
        [relu, relu6, prelu, lrelu, swish]
    """

    def __init__(
        self, expanded_channels: int, squeezed_channels: int, act_type: str = "relu"
    ):
        super().__init__()
        self.squeeze = AdaptiveAvgPool2d(1)
        self.reduce = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=expanded_channels,
                            out_channels=squeezed_channels,
                            kernel_size=1,
                        ),
                    ),
                    (
                        "act",
                        create_activation(
                            act_type, inplace=False, num_channels=squeezed_channels
                        ),
                    ),
                ]
            )
        )
        self.expand = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=squeezed_channels,
                            out_channels=expanded_channels,
                            kernel_size=1,
                        ),
                    ),
                    ("act", Sigmoid()),
                ]
            )
        )

    def forward(self, inp: Tensor):
        out = self.squeeze(inp)
        out = self.reduce(out)
        out = self.expand(out)

        return out
