from typing import List, Union
from torch import Tensor
from torch.nn import (
    Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, init, Sequential, Softmax
)

from .utils import load_pretrained_model, MODEL_MAPPINGS


__all__ = ['ResNetSectionSettings', 'ResNet',
           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet50_2xwidth', 'resnet101_2xwidth',
           'resnext50', 'resnext101', 'resnext152']


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
        self.conv = Conv2d(_Input.IN_CHANNELS, _Input.OUT_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = BatchNorm2d(_Input.OUT_CHANNELS)
        self.act = ReLU(inplace=True)
        self.pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)

        return out

    def initialize(self):
        _init_conv(self.conv)
        _init_batch_norm(self.bn)


class _IdentityModifier(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_channels)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)

        return out

    def initialize(self):
        _init_conv(self.conv)
        _init_batch_norm(self.bn)

    @staticmethod
    def required(in_channels: int, out_channels: int, stride: int) -> bool:
        return in_channels != out_channels or stride > 1


class _BasicBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.act1 = ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.identity = _IdentityModifier(in_channels, out_channels, stride) \
            if _IdentityModifier.required(in_channels, out_channels, stride) else None
        self.act_out = ReLU(inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity is not None:
            identity = self.identity(inp)
            out += identity
        else:
            out += inp

        out = self.act_out(out)

        return out

    def initialize(self):
        _init_conv(self.conv1)
        _init_batch_norm(self.bn1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn2, 0.0)


class _BottleneckBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, proj_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()

        self.conv1 = Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(proj_channels)
        self.act1 = ReLU(inplace=True)
        self.conv2 = Conv2d(proj_channels, proj_channels, kernel_size=3, stride=stride,
                            padding=1, bias=False, groups=groups)
        self.bn2 = BatchNorm2d(proj_channels)
        self.act2 = ReLU(inplace=True)
        self.conv3 = Conv2d(proj_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels)
        self.identity = _IdentityModifier(in_channels, out_channels, stride) \
            if _IdentityModifier.required(in_channels, out_channels, stride) else None
        self.act_out = ReLU(inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity is not None:
            identity = self.identity(inp)
            out += identity
        else:
            out += inp

        out = self.act_out(out)

        return out

    def initialize(self):
        _init_conv(self.conv1)
        _init_batch_norm(self.bn1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn2)
        _init_conv(self.conv3)
        _init_batch_norm(self.bn3, 0.0)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(in_channels, num_classes)
        self.softmax = Softmax(dim=1)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.avgpool(inp)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes

    def initialize(self):
        _init_linear(self.fc)


class ResNetSectionSettings(object):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, downsample: bool,
                 proj_channels: int = -1, groups: int = 1, use_se: bool = False):
        """
        :param num_blocks: the number of blocks to put in the section (ie Basic or Bottleneck blocks)
        :param in_channels: the number of input channels to the section
        :param out_channels: the number of output channels from the section
        :param downsample: True to apply stride 2 for downsampling of the input, False otherwise
        :param proj_channels: The number of channels in the projection for a bottleneck block, if < 0 then uses basic
        :param groups: The number of groups to use for each 3x3 conv (resnext)
        :param use_se: True to use squeeze excite, False otherwise
        """

        if use_se:
            # TODO: add support for squeeze excite
            raise NotImplementedError('squeeze excite not supported yet')

        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.proj_channels = proj_channels
        self.groups = groups
        self.use_se = use_se


class ResNet(Module):
    def __init__(self, sec_settings: List[ResNetSectionSettings], model_arch_tag: str, num_classes: int = 1000,
                 pretrained: Union[bool, str] = False):
        """
        Standard ResNet model
        https://arxiv.org/abs/1512.03385

        Also includes ResNext models
        https://arxiv.org/abs/1611.05431

        :param sec_settings: the settings for each section in the resnet model
        :param model_arch_tag: the architecture tag used for loading pretrained weights: ex resnet/50
        :param num_classes: the number of classes to classify
        :param pretrained: True to load dense, pretrained weights from imagenet, false otherwise
                           Additionally can specify other available datsets (dataset/dense) and
                           kernel sparsity models (dataset/sparse, dataset/sparse-perf)
        """
        super().__init__()
        self.input = _Input()
        self.sections = Sequential(*[ResNet.create_section(settings) for settings in sec_settings])
        self.classifier = _Classifier(sec_settings[-1].out_channels, num_classes)

        if pretrained:
            pretrained_key = pretrained if isinstance(pretrained, str) else ''
            load_pretrained_model(self, pretrained_key, model_arch=model_arch_tag,
                                  ignore_tensors=None if num_classes == 1000 else ['classifier.fc.weight',
                                                                                   'classifier.fc.bias'])

    def forward(self, inp: Tensor):
        out = self.input(inp)
        out = self.sections(out)
        logits, classes = self.classifier(out)

        return logits, classes

    @staticmethod
    def create_section(settings: ResNetSectionSettings) -> Sequential:
        blocks = []
        in_channels = settings.in_channels
        stride = 2 if settings.downsample else 1

        for _ in range(settings.num_blocks):
            if settings.proj_channels > 0:
                blocks.append(_BottleneckBlock(in_channels, settings.out_channels, settings.proj_channels,
                                               stride, settings.groups))
            else:
                blocks.append(_BasicBlock(in_channels, settings.out_channels, stride))

            in_channels = settings.out_channels
            stride = 1

        return Sequential(*blocks)


def resnet18(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=2, in_channels=64, out_channels=64, downsample=False),
        ResNetSectionSettings(num_blocks=2, in_channels=64, out_channels=128, downsample=True),
        ResNetSectionSettings(num_blocks=2, in_channels=128, out_channels=256, downsample=True),
        ResNetSectionSettings(num_blocks=2, in_channels=256, out_channels=512, downsample=True)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/18', **kwargs)


MODEL_MAPPINGS['resnet/18'] = resnet18


def resnet34(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=64, downsample=False),
        ResNetSectionSettings(num_blocks=4, in_channels=64, out_channels=128, downsample=True),
        ResNetSectionSettings(num_blocks=6, in_channels=128, out_channels=256, downsample=True),
        ResNetSectionSettings(num_blocks=3, in_channels=256, out_channels=512, downsample=True)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/34', **kwargs)


MODEL_MAPPINGS['resnet/34'] = resnet34


def resnet50(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False, proj_channels=64),
        ResNetSectionSettings(num_blocks=4, in_channels=256, out_channels=512, downsample=True, proj_channels=128),
        ResNetSectionSettings(num_blocks=6, in_channels=512, out_channels=1024, downsample=True, proj_channels=256),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True, proj_channels=512)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/50', **kwargs)


MODEL_MAPPINGS['resnet/50'] = resnet50


def resnet50_2xwidth(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False, proj_channels=128),
        ResNetSectionSettings(num_blocks=4, in_channels=256, out_channels=512, downsample=True, proj_channels=256),
        ResNetSectionSettings(num_blocks=6, in_channels=512, out_channels=1024, downsample=True, proj_channels=512),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True, proj_channels=1024)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/50_2xwidth', **kwargs)


MODEL_MAPPINGS['resnet/50_2xwidth'] = resnet50_2xwidth


def resnext50(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False,
                              proj_channels=128, groups=32),
        ResNetSectionSettings(num_blocks=4, in_channels=256, out_channels=512, downsample=True,
                              proj_channels=256, groups=32),
        ResNetSectionSettings(num_blocks=6, in_channels=512, out_channels=1024, downsample=True,
                              proj_channels=512, groups=32),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True,
                              proj_channels=1024, groups=32)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnext/50', **kwargs)


MODEL_MAPPINGS['resnext/50'] = resnext50


def resnet101(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False, proj_channels=64),
        ResNetSectionSettings(num_blocks=4, in_channels=256, out_channels=512, downsample=True, proj_channels=128),
        ResNetSectionSettings(num_blocks=23, in_channels=512, out_channels=1024, downsample=True, proj_channels=256),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True, proj_channels=512)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/101', **kwargs)


MODEL_MAPPINGS['resnet/101'] = resnet101


def resnet101_2xwidth(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False, proj_channels=128),
        ResNetSectionSettings(num_blocks=4, in_channels=256, out_channels=512, downsample=True, proj_channels=256),
        ResNetSectionSettings(num_blocks=23, in_channels=512, out_channels=1024, downsample=True, proj_channels=512),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True, proj_channels=1024)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/101_2xwidth', **kwargs)


MODEL_MAPPINGS['resnet/101_2xwidth'] = resnet101_2xwidth


def resnext101(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False,
                              proj_channels=128, groups=32),
        ResNetSectionSettings(num_blocks=4, in_channels=256, out_channels=512, downsample=True,
                              proj_channels=256, groups=32),
        ResNetSectionSettings(num_blocks=23, in_channels=512, out_channels=1024, downsample=True,
                              proj_channels=512, groups=32),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True,
                              proj_channels=1024, groups=32)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnext/101', **kwargs)


MODEL_MAPPINGS['resnext/101'] = resnext101


def resnet152(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False, proj_channels=64),
        ResNetSectionSettings(num_blocks=8, in_channels=256, out_channels=512, downsample=True, proj_channels=128),
        ResNetSectionSettings(num_blocks=36, in_channels=512, out_channels=1024, downsample=True, proj_channels=256),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True, proj_channels=512)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnet/152', **kwargs)


MODEL_MAPPINGS['resnet/152'] = resnet152


def resnext152(**kwargs) -> ResNet:
    sec_settings = [
        ResNetSectionSettings(num_blocks=3, in_channels=64, out_channels=256, downsample=False,
                              proj_channels=128, groups=32),
        ResNetSectionSettings(num_blocks=8, in_channels=256, out_channels=512, downsample=True,
                              proj_channels=256, groups=32),
        ResNetSectionSettings(num_blocks=36, in_channels=512, out_channels=1024, downsample=True,
                              proj_channels=512, groups=32),
        ResNetSectionSettings(num_blocks=3, in_channels=1024, out_channels=2048, downsample=True,
                              proj_channels=1024, groups=32)
    ]

    return ResNet(sec_settings=sec_settings, model_arch_tag='resnext/152', **kwargs)


MODEL_MAPPINGS['resnext/152'] = resnext152
