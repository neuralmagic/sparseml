# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch MobileNet V2 implementations.
Further info can be found in the paper `here <https://arxiv.org/abs/1801.04381>`__.
"""

from collections import OrderedDict
from typing import Dict, List, Union

from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    Module,
    Sequential,
    Sigmoid,
    Softmax,
    init,
)

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import ReLU6


__all__ = [
    "MobilenetV2SectionSettings",
    "MobilenetV2",
    "mobilenet_v2_width",
    "mobilenet_v2",
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
        expand_kwargs: Dict = None,
    ):
        super().__init__()

        if expand_kwargs is None:
            expand_kwargs = {"kernel_size": 1, "padding": 0, "stride": 1}

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
    """
    Settings to describe how to put together MobileNet V2 architecture
    using user supplied configurations.

    :param num_blocks: the number of inverted bottleneck blocks to put in the section
    :param in_channels: the number of input channels to the section
    :param out_channels: the number of output channels from the section
    :param downsample: True to apply stride 2 for down sampling of the input,
        False otherwise
    :param exp_channels: number of channels to expand out to,
        if not supplied uses exp_ratio
    :param exp_ratio: the expansion ratio to use for the depthwise convolution
    :param init_section: True if it is the initial section, False otherwise
    :param width_mult: The width multiplier to apply to the channel sizes
    """

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
    """
    Standard MobileNetV2 model https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self,
        sec_settings: List[MobilenetV2SectionSettings],
        num_classes: int,
        class_type: str,
    ):
        """
        :param sec_settings: the settings for each section in the mobilenet model
        :param num_classes: the number of classes to classify
        :param class_type: one of [single, multi] to support multi class training;
            default single
        """
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


def mobilenet_v2_width(
    width_mult: float, num_classes: int = 1000, class_type: str = "single"
) -> MobilenetV2:
    """
    Standard MobileNet V2 implementation for a width multiplier;
    expected input shape is (B, 3, 224, 224)

    :param width_mult: the width multiplier to apply
    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
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

    return MobilenetV2(sec_settings, num_classes, class_type)


@ModelRegistry.register(
    key=[
        "mobilenetv2",
        "mobilenet_v2",
        "mobilenet_v2_100",
        "mobilenet-v2",
        "mobilenet-v2-100",
        "mobilenetv2_1.0",
    ],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="mobilenet_v2",
    sub_architecture="1.0",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def mobilenet_v2(num_classes: int = 1000, class_type: str = "single") -> MobilenetV2:
    """
    Standard MobileNet V2 implementation for a width multiplier;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    return mobilenet_v2_width(
        width_mult=1.0, num_classes=num_classes, class_type=class_type
    )
