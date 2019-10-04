import math

from collections import OrderedDict
from typing import Union, List, Tuple, Dict

from torch import Tensor
from torch.nn import (
    Module, Sequential, Conv2d, BatchNorm2d, AdaptiveAvgPool2d, Sigmoid, Linear, Dropout, Softmax, ReLU)

from ..nn import Swish, SqueezeExcite
from .utils import load_pretrained_model, MODEL_MAPPINGS

__all__ = [
    'EfficientNet',
    'EfficientNetSectionSettings',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
]


class _EfficientNetConvBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, expansion_ratio: int, stride: int,
                 se_ratio: Union[float, None]):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        expanded_channels = in_channels * expansion_ratio

        self.expand = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, bias=False)),
            ('bn', BatchNorm2d(num_features=expanded_channels)),
            ('act', ReLU(inplace=True))
        ])) if expanded_channels != in_channels else None

        spatial_padding = (kernel_size - 1) // 2
        self.spatial = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=kernel_size,
                            stride=stride, padding=spatial_padding, groups=expanded_channels, bias=False)),
            ('bn', BatchNorm2d(num_features=expanded_channels)),
            ('act', ReLU(inplace=True))
        ]))

        squeezed_channels = max(1, int(in_channels * se_ratio)) if se_ratio and 0 < se_ratio <= 1 else None

        self.se = SqueezeExcite(expanded_channels, squeezed_channels) if squeezed_channels else None

        self.project = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels=expanded_channels, out_channels=out_channels, kernel_size=1, bias=False)),
            ('bn', BatchNorm2d(num_features=out_channels))
        ]))

    def forward(self, inp: Tensor):
        out = inp

        if self.expand is not None:
            out = self.expand(inp)

        out = self.spatial(out)

        if self.se is not None:
            out = out * self.se(out)

        out = self.project(out)

        if self._stride == 1 and self._in_channels == self._out_channels:
            out = out + inp

        return out


class _EfficientNetClassifier(Module):
    def __init__(self, in_channels: int, out_channels: int, classes: int, dropout: float = 0.0):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn = BatchNorm2d(num_features=out_channels)
        self.act = ReLU(inplace=True)
        self.pool = AdaptiveAvgPool2d(1)
        self.dropout = Dropout(p=dropout)
        self.fc = Linear(out_channels, classes)
        self.softmax = Softmax(dim=1)

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        # TODO: for nm-engine - pattern matches this
        out = self.pool(out)
        out = out.view(out.size(0), -1)

        # TODO: for mxnet-onnx - it cannot parse "concat0", which is why mean(..) is used here instead
        # out = out.mean(dim=2).mean(dim=2)

        out = self.dropout(out)
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes


class EfficientNetSectionSettings(object):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, kernel_size: int, expansion_ratio: int,
                 stride: int, se_ratio: Union[float, None]):
        """
        :param num_blocks: the number of blocks to put in the section
        :param in_channels: the number of input channels to the section
        :param out_channels: the number of output channels from the section
        :param kernel_size: the kernel size of the depth-wise convolution
        :param expansion_ratio: (in_channels * expansion_ratio) is the number of input/output channels of
               the depth-wise convolution
        :param stride: the stride of the depth-wise convolution
        :param se_ratio: (in_channels * se_ratio) is the number of input channels for squeeze-excite
        """

        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self.se_ratio = se_ratio


class EfficientNet(Module):
    def __init__(self, sec_settings: List[EfficientNetSectionSettings], model_arch_tag: str, num_classes: int,
                 out_channels: int, dropout: float = 0.0, pretrained: Union[bool, str]=False):
        super().__init__()
        self.input = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels=3, out_channels=sec_settings[0].in_channels,
                            kernel_size=3, stride=2, bias=False)),
            ('bn', BatchNorm2d(num_features=sec_settings[0].in_channels)),
            ('act', ReLU(inplace=True))
        ]))
        sections = []

        self.sections = Sequential(*[EfficientNet.create_section(settings) for settings in sec_settings])

        self.classifier = _EfficientNetClassifier(
            in_channels=sec_settings[-1].out_channels, out_channels=out_channels,
            classes=num_classes, dropout=dropout)

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.input(inp)
        feat = self.sections(feat)
        logits, classes = self.classifier(feat)

        return logits, classes

    @staticmethod
    def create_section(settings: EfficientNetSectionSettings) -> Sequential:
        assert settings.num_blocks > 0
        blocks = [_EfficientNetConvBlock(
            in_channels=settings.in_channels, out_channels=settings.out_channels,
            kernel_size=settings.kernel_size, expansion_ratio=settings.expansion_ratio,
            stride=settings.stride, se_ratio=settings.se_ratio)]

        for _ in range(settings.num_blocks - 1):
            blocks.append(_EfficientNetConvBlock(
                in_channels=settings.out_channels,
                out_channels=settings.out_channels,
                kernel_size=settings.kernel_size,
                expansion_ratio=settings.expansion_ratio, stride=1,
                se_ratio=settings.se_ratio))

        return Sequential(*blocks)


def _scale_num_channels(channels: int, width_mult: float) -> int:
    divisor = 8
    scaled = channels * width_mult
    scaled = max(divisor, int(scaled + divisor / 2) // divisor * divisor)

    if scaled < 0.9 * channels:
        # prevent rounding by more than 10%
        scaled += divisor

    return int(scaled)


def _scale_num_blocks(blocks: int, depth_mult: float) -> int:
    scaled = int(math.ceil(depth_mult * blocks))

    return scaled


def _create_section_settings(width_mult: float, depth_mult: float) -> List[EfficientNetSectionSettings]:
    return [
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(1, depth_mult),
            in_channels=_scale_num_channels(32, width_mult),
            out_channels=_scale_num_channels(16, width_mult),
            kernel_size=3,
            expansion_ratio=1,
            stride=1,
            se_ratio=0.25),
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(2, depth_mult),
            in_channels=_scale_num_channels(16, width_mult),
            out_channels=_scale_num_channels(24, width_mult),
            kernel_size=3,
            expansion_ratio=6,
            stride=2, se_ratio=0.25),
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(2, depth_mult),
            in_channels=_scale_num_channels(24, width_mult),
            out_channels=_scale_num_channels(40, width_mult),
            kernel_size=5,
            expansion_ratio=6,
            stride=2,
            se_ratio=0.25),
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(3, depth_mult),
            in_channels=_scale_num_channels(40, width_mult),
            out_channels=_scale_num_channels(80, width_mult),
            kernel_size=3,
            expansion_ratio=6,
            stride=2,
            se_ratio=0.25),
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(3, depth_mult),
            in_channels=_scale_num_channels(80, width_mult),
            out_channels=_scale_num_channels(112, width_mult),
            kernel_size=5,
            expansion_ratio=6,
            stride=1, se_ratio=0.25),
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(4, depth_mult),
            in_channels=_scale_num_channels(112, width_mult),
            out_channels=_scale_num_channels(192, width_mult),
            kernel_size=5,
            expansion_ratio=6,
            stride=2, se_ratio=0.25),
        EfficientNetSectionSettings(
            num_blocks=_scale_num_blocks(1, depth_mult),
            in_channels=_scale_num_channels(192, width_mult),
            out_channels=_scale_num_channels(320, width_mult),
            kernel_size=3,
            expansion_ratio=6,
            stride=1,
            se_ratio=0.25),
    ]


def _base_efficientnet_params(width_mult: float, depth_mult: float, dropout: float,
                              kwargs: Dict) -> Tuple[List[EfficientNetSectionSettings], Dict]:
    section_settings = _create_section_settings(width_mult, depth_mult)

    if 'out_channels' not in kwargs:
        kwargs['out_channels'] = _scale_num_channels(1280, width_mult)

    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 1000

    if 'dropout' not in kwargs:
        kwargs['dropout'] = dropout

    return section_settings, kwargs


def efficientnet_params(model_name):
    # Coefficients: width, depth, dropout, in_size
    params_dict = {
        'efficientnet_b0': (1.0, 1.0, 0.2, 224),
        'efficientnet_b1': (1.0, 1.1, 0.2, 240),
        'efficientnet_b2': (1.1, 1.2, 0.3, 260),
        'efficientnet_b3': (1.2, 1.4, 0.3, 300),
        'efficientnet_b4': (1.4, 1.8, 0.4, 380),
        'efficientnet_b5': (1.6, 2.2, 0.4, 456),
        'efficientnet_b6': (1.8, 2.6, 0.5, 528),
        'efficientnet_b7': (2.0, 3.1, 0.5, 600),
    }
    return params_dict[model_name]


def efficientnet_b0(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b0')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)

    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b0', **kwargs)


MODEL_MAPPINGS['efficientnet_b0'] = efficientnet_b0


def efficientnet_b1(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b1')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b1', **kwargs)


MODEL_MAPPINGS['efficientnet_b1'] = efficientnet_b1


def efficientnet_b2(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b2')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b2', **kwargs)


MODEL_MAPPINGS['efficientnet_b2'] = efficientnet_b2


def efficientnet_b3(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b3')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b3', **kwargs)


MODEL_MAPPINGS['efficientnet_b3'] = efficientnet_b3


def efficientnet_b4(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b4')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b4', **kwargs)


MODEL_MAPPINGS['efficientnet_b4'] = efficientnet_b4


def efficientnet_b5(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b5')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b5', **kwargs)


MODEL_MAPPINGS['efficientnet_b5'] = efficientnet_b5


def efficientnet_b6(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b6')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b6', **kwargs)


MODEL_MAPPINGS['efficientnet_b6'] = efficientnet_b6


def efficientnet_b7(**kwargs) -> EfficientNet:
    width_mult, depth_mult, dropout, _ = efficientnet_params('efficientnet_b7')

    sec_settings, kwargs = _base_efficientnet_params(width_mult, depth_mult, dropout, kwargs)
    return EfficientNet(sec_settings=sec_settings, model_arch_tag='efficientnet/b7', **kwargs)


MODEL_MAPPINGS['efficientnet_b7'] = efficientnet_b7
