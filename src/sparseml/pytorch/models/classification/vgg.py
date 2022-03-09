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
PyTorch VGG implementations.
Further info can be found in the paper `here <https://arxiv.org/abs/1409.1556>`__.
"""

from typing import List

from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    Sequential,
    Sigmoid,
    Softmax,
    init,
)

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import ReLU


__all__ = [
    "VGG",
    "vgg11",
    "vgg11bn",
    "vgg13",
    "vgg13bn",
    "vgg16",
    "vgg16bn",
    "vgg19bn",
    "vgg19",
]


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

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
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1
        )
        self.bn = BatchNorm2d(out_channels) if batch_norm else None
        self.act = ReLU(num_channels=out_channels, inplace=True)

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
    def __init__(self, in_channels: int, num_classes: int, class_type: str = "single"):
        super().__init__()
        self.mlp = Sequential(
            Linear(in_channels * 7 * 7, 4096),
            Dropout(),
            ReLU(num_channels=4096, inplace=True),
            Linear(4096, 4096),
            Dropout(),
            ReLU(num_channels=4096, inplace=True),
            Linear(4096, num_classes),
        )

        if class_type == "single":
            self.softmax = Softmax(dim=1)
        elif class_type == "multi":
            self.softmax = Sigmoid()
        else:
            raise ValueError("unknown class_type given of {}".format(class_type))

    def forward(self, inp: Tensor):
        out = inp.view(inp.size(0), -1)
        logits = self.mlp(out)
        classes = self.softmax(logits)

        return logits, classes


class VGGSectionSettings(object):
    """
    Settings to describe how to put together a VGG architecture
    using user supplied configurations.

    :param num_blocks: the number of blocks to put in the section (conv [bn] relu)
    :param in_channels: the number of input channels to the section
    :param out_channels: the number of output channels from the section
    :param use_batchnorm: True to put batchnorm after each conv, False otherwise
    """

    def __init__(
        self, num_blocks: int, in_channels: int, out_channels: int, use_batchnorm: bool
    ):
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm


class VGG(Module):
    """
    VGG implementation

    :param sec_settings: the settings for each section in the vgg model
    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    """

    def __init__(
        self,
        sec_settings: List[VGGSectionSettings],
        num_classes: int,
        class_type: str,
    ):
        super(VGG, self).__init__()
        self.sections = Sequential(
            *[VGG.create_section(settings) for settings in sec_settings]
        )
        self.classifier = _Classifier(
            sec_settings[-1].out_channels, num_classes, class_type
        )

    def forward(self, inp):
        out = self.sections(inp)
        logits, classes = self.classifier(out)

        return logits, classes

    @staticmethod
    def create_section(settings: VGGSectionSettings) -> Sequential:
        blocks = []
        in_channels = settings.in_channels

        for _ in range(settings.num_blocks):
            blocks.append(
                _Block(in_channels, settings.out_channels, settings.use_batchnorm)
            )
            in_channels = settings.out_channels

        blocks.append(MaxPool2d(kernel_size=2, stride=2))

        return Sequential(*blocks)


@ModelRegistry.register(
    key=["vgg11", "vgg_11", "vgg-11"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="11",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg11(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    Standard VGG 11; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=1, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=1, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg11bn", "vgg_11bn", "vgg-11bn"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="11_bn",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg11bn(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    VGG 11 with batch norm added; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=1, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=1, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg13", "vgg_13", "vgg-13"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="13",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg13(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    Standard VGG 13; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg13bn", "vgg_13bn", "vgg-13bn"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="13_bn",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg13bn(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    VGG 13 with batch norm added; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg16", "vgg_16", "vgg-16"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="16",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg16(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    Standard VGG 16; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg16bn", "vgg_16bn", "vgg-16bn"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="16_bn",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg16bn(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    VGG 16 with batch norm added; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=3, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg19", "vgg_19", "vgg-19"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="19",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg19(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    Standard VGG 19; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=128, out_channels=256, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=256, out_channels=512, use_batchnorm=False
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=512, out_channels=512, use_batchnorm=False
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["vgg19bn", "vgg_19bn", "vgg-19bn"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="19_bn",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.mlp.6.weight", "classifier.mlp.6.bias"],
)
def vgg19bn(num_classes: int = 1000, class_type: str = "single") -> VGG:
    """
    VGG 19 with batch norm added; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MobileNet Module
    """
    sec_settings = [
        VGGSectionSettings(
            num_blocks=2, in_channels=3, out_channels=64, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=128, out_channels=256, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=256, out_channels=512, use_batchnorm=True
        ),
        VGGSectionSettings(
            num_blocks=4, in_channels=512, out_channels=512, use_batchnorm=True
        ),
    ]

    return VGG(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )
