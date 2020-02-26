from typing import List, Union, Dict
from collections import OrderedDict
from torch import Tensor
from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    AdaptiveAvgPool2d,
    Linear,
    init,
    Sequential,
    Softmax,
    Sigmoid,
    Dropout,
)

from neuralmagicML.pytorch.nn import ReLU6
from neuralmagicML.pytorch.models.utils import load_pretrained_model, MODEL_MAPPINGS


__all__ = [
    "MobilenetV2SectionSettings",
    "MobilenetV2",
    "mobilenet_v2",
    "mobilenet_v2_100",
]


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")


def _init_batch_norm(norm: BatchNorm2d, weight_const: float = 1.0):
    init.constant_(norm.weight, weight_const)
    init.constant_(norm.bias, 0.0)


def _init_linear(linear: Linear):
    init.normal_(linear.weight, 0, 0.01)
    init.constant_(linear.bias, 0)


def _make_divisible(
    value: float, divisor: int, min_value: Union[int, None] = None
) -> int:
    # taken from the original implementation in tf:
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    if new_value < 0.9 * value:
        new_value += divisor

    return new_value


class _InvertedResidualBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_channels: int,
        stride: int,
        expand_kwargs: Dict = {"kernel_size": 1, "padding": 0, "stride": 1},
    ):
        super().__init__()
        self.expand = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(in_channels, exp_channels, bias=False, **expand_kwargs),
                    ),
                    ("bn", BatchNorm2d(exp_channels)),
                    ("act", ReLU6(num_channels=exp_channels, inplace=True)),
                ]
            )
        )
        self.spatial = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            exp_channels,
                            exp_channels,
                            kernel_size=3,
                            padding=1,
                            stride=stride,
                            groups=exp_channels,
                            bias=False,
                        ),
                    ),
                    ("bn", BatchNorm2d(exp_channels)),
                    ("act", ReLU6(num_channels=exp_channels, inplace=True)),
                ]
            )
        )
        self.compress = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
                    ),
                    ("bn", BatchNorm2d(out_channels)),
                ]
            )
        )
        self.include_identity = in_channels == out_channels and stride == 1

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.expand(inp)
        out = self.spatial(out)
        out = self.compress(out)

        if self.include_identity:
            out += inp

        return out

    def initialize(self):
        _init_conv(self.expand.conv)
        _init_batch_norm(self.expand.bn)
        _init_conv(self.spatial.conv)
        _init_batch_norm(self.spatial.bn)
        _init_conv(self.compress.conv)
        _init_batch_norm(self.compress.bn)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int, class_type: str = "single"):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.dropout = Dropout(0.2)
        self.fc = Linear(in_channels, num_classes)

        if class_type == "single":
            self.softmax = Softmax(dim=1)
        elif class_type == "multi":
            self.softmax = Sigmoid()
        else:
            raise ValueError("unknown class_type given of {}".format(class_type))

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.avgpool(inp)
        out = out.view(out.size(0), -1)
        logits = self.dropout(out)
        logits = self.fc(logits)
        classes = self.softmax(logits)

        return logits, classes

    def initialize(self):
        _init_linear(self.fc)


class MobilenetV2SectionSettings(object):
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        exp_channels: Union[None, int] = None,
        exp_ratio: float = 1.0,
        init_section: bool = False,
        width_mult: float = 1.0,
    ):
        self.num_blocks = num_blocks
        self.in_channels = (
            _make_divisible(in_channels * width_mult, 8)
            if not init_section
            else in_channels
        )
        self.out_channels = _make_divisible(out_channels * width_mult, 8)

        if exp_channels is not None:
            self.init_exp_channels = exp_channels
            self.exp_channels = exp_channels
        else:
            self.init_exp_channels = _make_divisible(self.in_channels * exp_ratio, 8)
            self.exp_channels = _make_divisible(self.out_channels * exp_ratio, 8)

        self.downsample = downsample
        self.init_section = init_section


class MobilenetV2(Module):
    def __init__(
        self,
        sec_settings: List[MobilenetV2SectionSettings],
        model_arch_tag: str,
        num_classes: int = 1000,
        class_type: str = "single",
        pretrained: Union[bool, str] = False,
    ):
        super().__init__()
        self.sections = Sequential(
            *[MobilenetV2.create_section(settings) for settings in sec_settings]
        )
        self.feat_extraction = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=sec_settings[-1].out_channels,
                            out_channels=1280,
                            kernel_size=1,
                            bias=False,
                        ),
                    ),
                    ("bn", BatchNorm2d(1280)),
                    ("act", ReLU6(num_channels=1280, inplace=True)),
                ]
            )
        )
        self.classifier = _Classifier(
            in_channels=1280, num_classes=num_classes, class_type=class_type
        )

        if pretrained:
            pretrained_key = pretrained if isinstance(pretrained, str) else ""
            load_pretrained_model(
                self,
                pretrained_key,
                model_arch=model_arch_tag,
                ignore_tensors=None
                if num_classes == 1000
                else ["classifier.fc.weight", "classifier.fc.bias"],
            )

    def forward(self, inp: Tensor):
        out = self.sections(inp)
        out = self.feat_extraction(out)
        logits, classes = self.classifier(out)

        return logits, classes

    @staticmethod
    def create_section(settings: MobilenetV2SectionSettings) -> Sequential:
        blocks = []
        in_channels = settings.in_channels
        stride = 2 if settings.downsample else 1
        exp_channels = settings.init_exp_channels
        apply_exp_kwargs = settings.init_section

        for _ in range(settings.num_blocks):
            if apply_exp_kwargs:
                blocks.append(
                    _InvertedResidualBlock(
                        in_channels,
                        settings.out_channels,
                        exp_channels,
                        stride,
                        expand_kwargs={"kernel_size": 3, "padding": 1, "stride": 2},
                    )
                )
            else:
                blocks.append(
                    _InvertedResidualBlock(
                        in_channels, settings.out_channels, exp_channels, stride
                    )
                )

            in_channels = settings.out_channels
            exp_channels = settings.exp_channels
            stride = 1
            apply_exp_kwargs = False

        return Sequential(*blocks)


def mobilenet_v2(width_mult, model_arch_tag: str, **kwargs) -> MobilenetV2:
    sec_settings = [
        MobilenetV2SectionSettings(
            num_blocks=1,
            in_channels=3,
            out_channels=16,
            exp_channels=32,
            downsample=False,
            init_section=True,
            width_mult=width_mult,
        ),
        MobilenetV2SectionSettings(
            num_blocks=2,
            in_channels=16,
            out_channels=24,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobilenetV2SectionSettings(
            num_blocks=3,
            in_channels=24,
            out_channels=32,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobilenetV2SectionSettings(
            num_blocks=4,
            in_channels=32,
            out_channels=64,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobilenetV2SectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=96,
            exp_ratio=6,
            downsample=False,
            init_section=False,
            width_mult=width_mult,
        ),
        MobilenetV2SectionSettings(
            num_blocks=3,
            in_channels=96,
            out_channels=160,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobilenetV2SectionSettings(
            num_blocks=1,
            in_channels=160,
            out_channels=320,
            exp_ratio=6,
            downsample=False,
            init_section=False,
            width_mult=width_mult,
        ),
    ]

    return MobilenetV2(sec_settings, model_arch_tag=model_arch_tag, **kwargs)


def mobilenet_v2_100(**kwargs) -> MobilenetV2:
    return mobilenet_v2(width_mult=1.0, model_arch_tag="mobilenetv2/1.0", **kwargs)


MODEL_MAPPINGS["mobilenet_v2_100"] = mobilenet_v2_100
