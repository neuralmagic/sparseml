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
PyTorch MobileNet implementations.
Further info can be found in the paper `here <https://arxiv.org/abs/1704.04861>`__.
"""

from typing import List, Union

from torch import Tensor
from torch.nn import (
    AvgPool2d,
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
from sparseml.pytorch.nn import ReLU


__all__ = ["MobileNetSectionSettings", "MobileNet", "mobilenet", "han_mobilenet"]


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")


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
        self.conv = Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
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
        self.depth = _ConvBNRelu(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.point = _ConvBNRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, inp: Tensor):
        out = self.depth(inp)
        out = self.point(out)

        return out


class _Classifier(Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        class_type: str = "single",
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.avgpool = AvgPool2d(7)
        self.fc = Linear(in_channels, num_classes)

        if dropout is not None:  # add dropout layer before fc
            self.fc = Sequential(Dropout(dropout), self.fc)

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
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes

    def initialize(self):
        fc = self.fc if not isinstance(self.fc, Sequential) else self.fc[1]
        _init_linear(fc)


class MobileNetSectionSettings(object):
    """
    Settings to describe how to put together a MobileNet architecture
    using user supplied configurations.

    :param num_blocks: the number of depthwise separable blocks to put in the section
    :param in_channels: the number of input channels to the section
    :param out_channels: the number of output channels from the section
    :param downsample: True to apply stride 2 for down sampling of the input,
        False otherwise
    """

    def __init__(
        self, num_blocks: int, in_channels: int, out_channels: int, downsample: bool
    ):
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample


class MobileNet(Module):
    """
    MobileNet implementation

    :param sec_settings: the settings for each section in the MobileNet model
    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: dropout level for input to FC layer; setting to None performs
        no dropout; default is None
    """

    def __init__(
        self,
        sec_settings: List[MobileNetSectionSettings],
        num_classes: int,
        class_type: str,
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.input = _Input()
        self.sections = Sequential(
            *[MobileNet.create_section(settings) for settings in sec_settings]
        )
        self.classifier = _Classifier(
            sec_settings[-1].out_channels, num_classes, class_type, dropout
        )

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


def _mobilenet_base_section_settings():
    return [
        MobileNetSectionSettings(
            num_blocks=1, in_channels=32, out_channels=64, downsample=False
        ),
        MobileNetSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, downsample=True
        ),
        MobileNetSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, downsample=True
        ),
        MobileNetSectionSettings(
            num_blocks=6, in_channels=256, out_channels=512, downsample=True
        ),
        MobileNetSectionSettings(
            num_blocks=2, in_channels=512, out_channels=1024, downsample=True
        ),
    ]


@ModelRegistry.register(
    key=[
        "mobilenet",
        "mobilenet_100",
        "mobilenet-v1",
        "mobilenet-v1-100",
        "mobilenet_v1",
        "mobilenet_v1_100",
        "mobilenetv1_1.0",
    ],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="mobilenet_v1",
    sub_architecture="1.0",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def mobilenet(num_classes: int = 1000, class_type: str = "single") -> MobileNet:
    """
    Standard MobileNet implementation with width=1.0;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = _mobilenet_base_section_settings()
    return MobileNet(sec_settings, num_classes, class_type)


@ModelRegistry.register(
    key=[
        "han-mobilenet",
        "han_mobilenet",
        "mobilenet-han",
        "mobilenet_han",
    ],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="mobilenet_v1",
    sub_architecture="han",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc[1].weight", "classifier.fc[1].bias"],
)
def han_mobilenet(num_classes: int = 1000, class_type: str = "single") -> MobileNet:
    """
    Standard MobileNet implementation with width=1.0;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = _mobilenet_base_section_settings()
    return MobileNet(sec_settings, num_classes, class_type, dropout=0.2)
