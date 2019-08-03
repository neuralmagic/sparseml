from typing import List
from torch import Tensor
from torch.nn import (
    Module, Conv2d, BatchNorm2d, AdaptiveAvgPool2d, MaxPool2d, Linear, init, Sequential, Dropout, Softmax, ReLU
)

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
]


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

    if conv.bias is not None:
        init.constant_(conv.bias, 0)


def _init_batch_norm(norm: BatchNorm2d):
    init.constant_(norm.weight, 1.0)
    init.constant_(norm.bias, 0.0)


def _init_linear(linear: Linear):
    init.normal_(linear.weight, 0, 0.01)
    init.constant_(linear.bias, 0)


class _Block(Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=not batch_norm)
        self.bn = BatchNorm2d(out_channels) if batch_norm else None
        self.act = ReLU(inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)

        if self.bn is not None:
            out = self.bn(out)

        out = self.act(out)

        return out

    def initialize(self):
        _init_conv(self.conv)

        if self.bn is not None:
            _init_batch_norm(self.bn)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(7)
        self.mlp = Sequential(
            Linear(in_channels * 7 * 7, 4096),
            Dropout(),
            ReLU(inplace=True),
            Linear(4096, 4096),
            Dropout(),
            ReLU(inplace=True),
            Linear(4096, num_classes)
        )
        self.softmax = Softmax(dim=1)

    def forward(self, inp: Tensor):
        out = self.avgpool(inp)
        out = out.view(inp.size(0), -1)
        logits = self.mlp(out)
        classes = self.softmax(logits)

        return logits, classes


class VGGSectionSettings(object):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, use_batchnorm: bool):
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm


class VGG(Module):
    def __init__(self, sec_settings: List[VGGSectionSettings], num_classes: int = 1000, pretrained: bool = False):
        super(VGG, self).__init__()
        self.sections = Sequential(*[VGG.create_section(settings) for settings in sec_settings])
        self.classifier = _Classifier(sec_settings[-1].out_channels, num_classes)

        if pretrained:
            # TODO: add loading of pretrained weights for initialization
            raise Exception('pretrained not currently supported')

    def forward(self, inp):
        out = self.sections(inp)
        logits, classes = self.classifier(out)

        return logits, classes

    @staticmethod
    def create_section(settings: VGGSectionSettings) -> Sequential:
        blocks = []
        in_channels = settings.in_channels

        for _ in range(settings.num_blocks):
            blocks.append(_Block(in_channels, settings.out_channels, settings.use_batchnorm))
            in_channels = settings.out_channels

        blocks.append(MaxPool2d(kernel_size=2, stride=2))

        return Sequential(*blocks)


def vgg11(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=1, in_channels=3, out_channels=64, use_batchnorm=False),
        VGGSectionSettings(num_blocks=1, in_channels=64, out_channels=128, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=False)
    ]

    return VGG(sec_settings, **kwargs)


def vgg11_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=1, in_channels=3, out_channels=64, use_batchnorm=True),
        VGGSectionSettings(num_blocks=1, in_channels=64, out_channels=128, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=True)
    ]

    return VGG(sec_settings, **kwargs)


def vgg13(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=False)
    ]

    return VGG(sec_settings, **kwargs)


def vgg13_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=True)
    ]

    return VGG(sec_settings, **kwargs)


def vgg16(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False),
        VGGSectionSettings(num_blocks=3, in_channels=128, out_channels=256, use_batchnorm=False),
        VGGSectionSettings(num_blocks=3, in_channels=256, out_channels=512, use_batchnorm=False),
        VGGSectionSettings(num_blocks=3, in_channels=512, out_channels=512, use_batchnorm=False)
    ]

    return VGG(sec_settings, **kwargs)


def vgg16_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True),
        VGGSectionSettings(num_blocks=3, in_channels=128, out_channels=256, use_batchnorm=True),
        VGGSectionSettings(num_blocks=3, in_channels=256, out_channels=512, use_batchnorm=True),
        VGGSectionSettings(num_blocks=3, in_channels=512, out_channels=512, use_batchnorm=True)
    ]

    return VGG(sec_settings, **kwargs)


def vgg19(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False),
        VGGSectionSettings(num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False),
        VGGSectionSettings(num_blocks=4, in_channels=128, out_channels=256, use_batchnorm=False),
        VGGSectionSettings(num_blocks=4, in_channels=256, out_channels=512, use_batchnorm=False),
        VGGSectionSettings(num_blocks=4, in_channels=512, out_channels=512, use_batchnorm=False)
    ]

    return VGG(sec_settings, **kwargs)


def vgg19_bn(**kwargs) -> VGG:
    sec_settings = [
        VGGSectionSettings(num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True),
        VGGSectionSettings(num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True),
        VGGSectionSettings(num_blocks=4, in_channels=128, out_channels=256, use_batchnorm=True),
        VGGSectionSettings(num_blocks=4, in_channels=256, out_channels=512, use_batchnorm=True),
        VGGSectionSettings(num_blocks=4, in_channels=512, out_channels=512, use_batchnorm=True)
    ]

    return VGG(sec_settings, **kwargs)
