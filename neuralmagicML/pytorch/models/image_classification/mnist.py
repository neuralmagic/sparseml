from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    AdaptiveAvgPool2d,
    Conv2d,
    BatchNorm2d,
    Linear,
    Softmax,
)

from neuralmagicML.pytorch.nn import ReLU
from neuralmagicML.pytorch.utils.model import MODEL_MAPPINGS, load_pretrained_model


__all__ = ["MnistNet", "mnist_net"]


class _ConvBNRelu(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False,
        )
        self.bn = BatchNorm2d(out_channels)
        self.act = ReLU(num_channels=out_channels, inplace=True)

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out


class _Classifier(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(in_channels, 10)
        self.softmax = Softmax(dim=1)

    def forward(self, inp: Tensor):
        out = self.avgpool(inp)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes


class MnistNet(Module):
    def __init__(
        self, pretrained: bool = False,
    ):
        """
        A simple convolutional model created for the MNIST dataset

        :param pretrained: True to load pretrained weights, false otherwise
        """
        super().__init__()
        self.blocks = Sequential(
            _ConvBNRelu(
                in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=1
            ),
            _ConvBNRelu(
                in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2
            ),
            _ConvBNRelu(
                in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1
            ),
            _ConvBNRelu(
                in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2
            ),
        )
        self.classifier = _Classifier(in_channels=128)

        if pretrained:
            pretrained_key = pretrained if isinstance(pretrained, str) else ""
            load_pretrained_model(
                self,
                pretrained_key,
                model_arch="mnistnet/none",
                default_pretrained_key="mnist/pytorch/base",
            )

    def forward(self, inp: Tensor):
        out = self.blocks(inp)
        logits, classes = self.classifier(out)

        return logits, classes


def mnist_net(**kwargs) -> MnistNet:
    return MnistNet(**kwargs)


MODEL_MAPPINGS["mnist_net"] = mnist_net
