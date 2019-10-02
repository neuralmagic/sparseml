from typing import List
from torch import Tensor
from torch.nn import Module, Sequential, AvgPool2d, Conv2d, BatchNorm2d, Linear, Softmax, Sigmoid, init

from ..nn import ReLU
from .utils import load_pretrained_model, MODEL_MAPPINGS


__all__ = ['MobileNetSectionSettings', 'MobileNet', 'mobilenet']


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')


def _init_batch_norm(norm: BatchNorm2d, weight_const: float = 1.0):
    init.constant_(norm.weight, weight_const)
    init.constant_(norm.bias, 0.0)


def _init_linear(linear: Linear):
    init.normal_(linear.weight, 0, 0.01)
    init.constant_(linear.bias, 0)


class _Input(Module):
    IN_CHANNELS = 3
    OUT_CHANNELS = 64

    def __init__(self):
        super().__init__()
        self.conv = Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = BatchNorm2d(32)
        self.act = ReLU(num_channels=32, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out

    def initialize(self):
        _init_conv(self.conv)
        _init_batch_norm(self.bn)


class _ConvBNRelu(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, groups: int):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.act = ReLU(num_channels=out_channels, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out

    def initialize(self):
        _init_conv(self.conv)
        _init_batch_norm(self.bn)


class _Block(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depth = _ConvBNRelu(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.point = _ConvBNRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, inp: Tensor):
        out = self.depth(inp)
        out = self.point(out)

        return out


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int, class_type: str = 'single'):
        super().__init__()
        self.avgpool = AvgPool2d(7)
        self.fc = Linear(in_channels, num_classes)

        if class_type == 'single':
            self.softmax = Softmax(dim=1)
        elif class_type == 'multi':
            self.softmax = Sigmoid()
        else:
            raise ValueError('unknown class_type given of {}'.format(class_type))

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.avgpool(inp)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes

    def initialize(self):
        _init_linear(self.fc)


class MobileNetSectionSettings(object):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, downsample: bool):
        """
        :param num_blocks: the number of depthwise separable blocks to put in the section
        :param in_channels: the number of input channels to the section
        :param out_channels: the number of output channels from the section
        :param downsample: True to apply stride 2 for downsampling of the input, False otherwise
        """
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample


class MobileNet(Module):
    def __init__(self, sec_settings: List[MobileNetSectionSettings], num_classes: int = 1000,
                 class_type: str = 'single', pretrained: bool = False):
        """
        Standard MobileNet model
        https://arxiv.org/abs/1704.04861

        :param sec_settings: the settings for each section in the mobilenet model
        :param num_classes: the number of classes to classify
        :param pretrained: True to load pretrained weights from imagenet, false otherwise
        """
        super().__init__()
        self.input = _Input()
        self.sections = Sequential(*[MobileNet.create_section(settings) for settings in sec_settings])
        self.classifier = _Classifier(sec_settings[-1].out_channels, num_classes, class_type)

        if pretrained:
            pretrained_key = pretrained if isinstance(pretrained, str) else ''
            load_pretrained_model(self, pretrained_key, model_arch='mobilenet/1.0',
                                  ignore_tensors=None if num_classes == 1000 else ['classifier.fc.weight',
                                                                                   'classifier.fc.bias'])

    def forward(self, inp: Tensor):
        out = self.input(inp)
        out = self.sections(out)
        logits, classes = self.classifier(out)

        return logits, classes

    @staticmethod
    def create_section(settings: MobileNetSectionSettings) -> Sequential:
        blocks = []

        in_channels = settings.in_channels
        stride = 2 if settings.downsample else 1

        for _ in range(settings.num_blocks):
            blocks.append(_Block(in_channels, settings.out_channels, stride))
            in_channels = settings.out_channels
            stride = 1

        return Sequential(*blocks)


def mobilenet(**kwargs) -> MobileNet:
    sec_settings = [
        MobileNetSectionSettings(num_blocks=1, in_channels=32, out_channels=64, downsample=False),
        MobileNetSectionSettings(num_blocks=2, in_channels=64, out_channels=128, downsample=True),
        MobileNetSectionSettings(num_blocks=2, in_channels=128, out_channels=256, downsample=True),
        MobileNetSectionSettings(num_blocks=6, in_channels=256, out_channels=512, downsample=True),
        MobileNetSectionSettings(num_blocks=2, in_channels=512, out_channels=1024, downsample=True)
    ]

    return MobileNet(sec_settings, **kwargs)


MODEL_MAPPINGS['mobilenet'] = mobilenet
