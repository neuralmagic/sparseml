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
DarkNet classification model for use as YOLO OD backbone
"""

from typing import List, Union

from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    Sequential,
    Sigmoid,
    Softmax,
    init,
)

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import Hardswish


__all__ = [
    "DarkNetSectionSettings",
    "DarkNet",
    "darknet53",
]


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
    HIDDEN_CHANNELS = 32
    OUT_CHANNELS = 64

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(
            _Input.IN_CHANNELS,
            _Input.HIDDEN_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = BatchNorm2d(_Input.HIDDEN_CHANNELS, momentum=0.03, eps=1e-4)
        self.act1 = Hardswish(num_channels=_Input.HIDDEN_CHANNELS, inplace=True)
        self.conv2 = Conv2d(
            _Input.HIDDEN_CHANNELS,
            _Input.OUT_CHANNELS,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = BatchNorm2d(_Input.OUT_CHANNELS, momentum=0.03, eps=1e-4)
        self.act2 = Hardswish(num_channels=_Input.OUT_CHANNELS, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        return out

    def initialize(self):
        _init_conv(self.conv1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn1)
        _init_batch_norm(self.bn2)


class _ResidualBlock(Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = BatchNorm2d(hidden_channels, momentum=0.03, eps=1e-4)
        self.act1 = Hardswish(num_channels=hidden_channels, inplace=True)
        self.conv2 = Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = BatchNorm2d(out_channels, momentum=0.03, eps=1e-4)
        self.act2 = Hardswish(num_channels=out_channels, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        # identity step, correct for channel mis-matches
        in_channels = inp.size(1)
        out_channels = out.size(1)
        if in_channels == out_channels:  # standard case
            out += inp
        elif in_channels < out_channels:
            out[:, :in_channels] += inp
        else:
            out += inp[:, :out_channels]

        return out

    def initialize(self):
        _init_conv(self.conv1)
        _init_batch_norm(self.bn1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn2, 0.0)


class _DownsampleBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels, momentum=0.03, eps=1e-4)
        self.act = Hardswish(num_channels=out_channels, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out

    def initialize(self):
        _init_conv(self.conv)
        _init_batch_norm(self.bn)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int, class_type: str = "single"):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(output_size=1)
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
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes

    def initialize(self):
        _init_linear(self.fc)


class DarkNetSectionSettings(object):
    """
    Settings to describe how to put together a DarkNet based architecture
    using user supplied configurations.

    :param num_blocks: the number of residual blocks to put in the section
    :param in_channels: the number of input channels to the section
    :param hidden_channels: the number of hidden channels in the residual blocks
    :param out_channels: the number of output channels for this sections residual blocks
    :param downsample_out_channels: number of output channels to apply to an additional
        convolution downsample layer. Setting to None will omit this layer. Default is
        None.
    """

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        downsample_out_channels: Union[int, None] = None,
    ):
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.downsample_out_channels = downsample_out_channels


class DarkNet(Module):
    """
    DarkNet implementation

    :param sec_settings: the settings for each section in the DarkNet model
    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    """

    def __init__(
        self,
        sec_settings: List[DarkNetSectionSettings],
        num_classes: int,
        class_type: str,
    ):
        super().__init__()
        self.input = _Input()
        self.sections = Sequential(
            *[DarkNet.create_section(settings) for settings in sec_settings]
        )
        self.classifier = _Classifier(
            sec_settings[-1].out_channels, num_classes, class_type
        )
        self.backbone_outputs = []

    def forward(self, inp: Tensor) -> Union[Tensor, List[Tensor]]:
        out = self.input(inp)

        if self.backbone_outputs:
            outputs = {}
            for idx, section in enumerate(self.sections):
                if idx not in self.backbone_outputs:
                    out = section(out)
                    continue
                if isinstance(section[-1], _DownsampleBlock):
                    for block in section[:-1]:
                        out = block(out)
                    outputs[idx] = out  # return residual output as backbone feature
                    out = section[-1](out)  # run downsample block
                else:
                    out = section(out)
                    outputs[idx] = out
            # return outputs in order of self.backbone_output indices
            return [outputs[sec_idx] for sec_idx in self.backbone_outputs]

        else:  # defaults to classifier
            out = self.sections(out)
            logits, classes = self.classifier(out)

        return logits, classes

    def as_yolo_backbone(self, output_blocks: List[int] = None):
        """
        Sets this model to output the given residual block indices as a backbone
        feature extractor for a detection model such as Yolo.

        :param output_blocks: indices of residual DarkNet blocks to output as backbone.
            Default is the final block output.
        """
        if output_blocks is None:
            output_blocks = [-1]
        for idx in output_blocks:
            if idx < -len(self.sections) or idx >= len(self.sections):
                raise ValueError(
                    "Index {} out of range for DarkNet model with {} sections".format(
                        idx, len(self.sections)
                    )
                )
        output_blocks = [  # convert negative indices
            idx if idx >= 0 else idx + len(output_blocks) for idx in output_blocks
        ]

        self.backbone_outputs = output_blocks

    def as_classifier(self):
        """
        Sets this model to return output as an image classifier through a final FC layer
        """
        self.backbone_outputs = []

    @staticmethod
    def create_section(settings: DarkNetSectionSettings) -> Sequential:
        blocks = []
        for _ in range(settings.num_blocks):
            blocks.append(
                _ResidualBlock(
                    settings.in_channels,
                    settings.hidden_channels,
                    settings.out_channels,
                )
            )
        if settings.downsample_out_channels is not None:
            blocks.append(
                _DownsampleBlock(
                    settings.out_channels, settings.downsample_out_channels
                )
            )
        return Sequential(*blocks)


@ModelRegistry.register(
    key=["darknet53", "darknet_53", "darknet-53"],
    input_shape=(3, 256, 256),
    domain="cv",
    sub_domain="classification",
    architecture="darknet",
    sub_architecture="53",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def darknet53(num_classes: int = 1000, class_type: str = "single") -> DarkNet:
    """
    DarkNet-53 implementation as described in the Yolo v3 paper;
    expected input shape is (B, 3, 256, 256)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created DarkNet Module
    """
    sec_settings = [
        DarkNetSectionSettings(1, 64, 32, 64, 128),
        DarkNetSectionSettings(2, 128, 64, 128, 256),
        DarkNetSectionSettings(8, 256, 128, 256, 512),
        DarkNetSectionSettings(8, 512, 256, 512, 1024),
        DarkNetSectionSettings(4, 1024, 512, 1024, None),
    ]
    return DarkNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )
