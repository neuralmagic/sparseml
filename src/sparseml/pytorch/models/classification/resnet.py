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
PyTorch ResNet, ResNet V2, ResNext implementations.
Further info on ResNet can be found in the paper
`here <https://arxiv.org/abs/1512.03385>`__.
Further info on ResNet V2 can be found in the paper
`here <https://arxiv.org/abs/1603.05027>`__.
Further info on ResNext can be found in the paper
`here <https://arxiv.org/abs/1611.05431>`__.
"""

from typing import List

from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
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


try:
    from torch.nn.quantized import FloatFunctional
except Exception:
    FloatFunctional = None

__all__ = [
    "ResNetSectionSettings",
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnetv2_18",
    "resnetv2_34",
    "resnetv2_50",
    "resnetv2_101",
    "resnetv2_152",
    "resnet50_2xwidth",
    "resnet101_2xwidth",
    "resnext50",
    "resnext101",
    "resnext152",
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
    OUT_CHANNELS = 64

    def __init__(self):
        super().__init__()
        self.conv = Conv2d(
            _Input.IN_CHANNELS,
            _Input.OUT_CHANNELS,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = BatchNorm2d(_Input.OUT_CHANNELS)
        self.act = ReLU(num_channels=_Input.OUT_CHANNELS, inplace=True)
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
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=False
        )
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


class _AddReLU(Module):
    """
    Wrapper for the FloatFunctional class that enables QATWrapper used to
    quantize the first input to the Add operation
    """

    def __init__(self, num_channels):
        super().__init__()
        if FloatFunctional:
            self.functional = FloatFunctional()
            self.wrap_qat = True
            self.qat_wrapper_kwargs = {"num_inputs": 1, "num_outputs": 1}
        else:
            self.functional = ReLU(num_channels=num_channels, inplace=True)

    def forward(self, x, y):
        if isinstance(self.functional, FloatFunctional):
            return self.functional.add_relu(x, y)
        else:
            return self.functional(x + y)


class _BasicBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = BatchNorm2d(out_channels)
        self.act1 = ReLU(num_channels=out_channels, inplace=True)
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(out_channels)
        self.identity = (
            _IdentityModifier(in_channels, out_channels, stride)
            if _IdentityModifier.required(in_channels, out_channels, stride)
            else None
        )

        # self.add_relu = _AddReLU(out_channels)
        if FloatFunctional:
            self.add_relu = FloatFunctional()
        else:
            self.add_relu = ReLU(num_channels=out_channels, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity_val = self.identity(inp) if self.identity is not None else inp
        # out = self.add_relu(identity_val, out)
        # return out

        if isinstance(self.add_relu, FloatFunctional):
            return self.add_relu.add_relu(identity_val, out)
        else:
            return self.add_relu(identity_val + out)

    def initialize(self):
        _init_conv(self.conv1)
        _init_batch_norm(self.bn1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn2, 0.0)


class _BottleneckBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        proj_channels: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        self.conv1 = Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(proj_channels)
        self.act1 = ReLU(num_channels=proj_channels, inplace=True)
        self.conv2 = Conv2d(
            proj_channels,
            proj_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = BatchNorm2d(proj_channels)
        self.act2 = ReLU(num_channels=proj_channels, inplace=True)
        self.conv3 = Conv2d(proj_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels)
        self.identity = (
            _IdentityModifier(in_channels, out_channels, stride)
            if _IdentityModifier.required(in_channels, out_channels, stride)
            else None
        )

        # self.add_relu = _AddReLU(out_channels)
        if FloatFunctional:
            self.add_relu = FloatFunctional()
        else:
            self.add_relu = ReLU(num_channels=out_channels, inplace=True)

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

        identity_val = self.identity(inp) if self.identity is not None else inp

        # out = self.add_relu(identity_val, out)
        # return out

        if isinstance(self.add_relu, FloatFunctional):
            return self.add_relu.add_relu(identity_val, out)
        else:
            return self.add_relu(identity_val + out)

    def initialize(self):
        _init_conv(self.conv1)
        _init_batch_norm(self.bn1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn2)
        _init_conv(self.conv3)
        _init_batch_norm(self.bn3, 0.0)


class _BasicBlockV2(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = BatchNorm2d(in_channels)
        self.act1 = ReLU(num_channels=in_channels, inplace=True)
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = BatchNorm2d(out_channels)
        self.act2 = ReLU(num_channels=out_channels, inplace=True)
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.identity = (
            Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else None
        )

        self.initialize()

    def forward(self, inp: Tensor):
        identity = inp

        out = self.bn1(inp)
        out = self.act1(out)
        if self.identity is not None:
            identity = self.identity(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out += identity

        return out

    def initialize(self):
        _init_conv(self.conv1)
        _init_batch_norm(self.bn1)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn2, 0.0)


class _BottleneckBlockV2(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        proj_channels: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        self.bn1 = BatchNorm2d(in_channels)
        self.act1 = ReLU(num_channels=in_channels, inplace=True)
        self.conv1 = Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)

        self.bn2 = BatchNorm2d(proj_channels)
        self.act2 = ReLU(num_channels=proj_channels, inplace=True)
        self.conv2 = Conv2d(
            proj_channels,
            proj_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.bn3 = BatchNorm2d(proj_channels)
        self.act3 = ReLU(num_channels=proj_channels, inplace=True)
        self.conv3 = Conv2d(proj_channels, out_channels, kernel_size=1, bias=False)

        self.identity = (
            Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else None
        )

        self.initialize()

    def forward(self, inp: Tensor):
        identity = inp

        out = self.bn1(inp)
        out = self.act1(out)
        if self.identity is not None:
            identity = self.identity(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)

        out += identity

        return out

    def initialize(self):
        _init_batch_norm(self.bn1)
        _init_conv(self.conv1)
        _init_batch_norm(self.bn2)
        _init_conv(self.conv2)
        _init_batch_norm(self.bn3)
        _init_conv(self.conv3)

        if self.identity is not None:
            _init_conv(self.identity)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int, class_type: str = "single"):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
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


class ResNetSectionSettings(object):
    """
    Settings to describe how to put together a ResNet based architecture
    using user supplied configurations.

    :param num_blocks: the number of blocks to put in the section
        (ie Basic or Bottleneck blocks)
    :param in_channels: the number of input channels to the section
    :param out_channels: the number of output channels from the section
    :param downsample: True to apply stride 2 for downsampling of the input,
        False otherwise
    :param proj_channels: The number of channels in the projection for a
        bottleneck block, if < 0 then uses basic
    :param groups: The number of groups to use for each 3x3 conv (ResNext)
    :param use_se: True to use squeeze excite, False otherwise
    :param version: 1 for original ResNet model, 2 for ResNet v2 model
    """

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        proj_channels: int = -1,
        groups: int = 1,
        use_se: bool = False,
        version: int = 1,
    ):
        if use_se:
            # TODO: add support for squeeze excite
            raise NotImplementedError("squeeze excite not supported yet")

        if version != 1 and version != 2:
            raise ValueError(
                "unknown version given of {}, only 1 and 2 are supported".format(
                    version
                )
            )

        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.proj_channels = proj_channels
        self.groups = groups
        self.use_se = use_se
        self.version = version


class ResNet(Module):
    """
    ResNet, ResNet V2, ResNext implementations.

    :param sec_settings: the settings for each section in the ResNet model
    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    """

    def __init__(
        self,
        sec_settings: List[ResNetSectionSettings],
        num_classes: int,
        class_type: str,
    ):
        super().__init__()
        self.input = _Input()
        self.sections = Sequential(
            *[ResNet.create_section(settings) for settings in sec_settings]
        )
        self.classifier = _Classifier(
            sec_settings[-1].out_channels, num_classes, class_type
        )

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
            if settings.proj_channels > 0 and settings.version == 1:
                blocks.append(
                    _BottleneckBlock(
                        in_channels,
                        settings.out_channels,
                        settings.proj_channels,
                        stride,
                        settings.groups,
                    )
                )
            elif settings.proj_channels > 0 and settings.version == 2:
                blocks.append(
                    _BottleneckBlockV2(
                        in_channels,
                        settings.out_channels,
                        settings.proj_channels,
                        stride,
                        settings.groups,
                    )
                )
            elif settings.version == 1:
                blocks.append(_BasicBlock(in_channels, settings.out_channels, stride))
            elif settings.version == 2:
                blocks.append(_BasicBlockV2(in_channels, settings.out_channels, stride))
            else:
                raise ValueError(
                    "could not figure out which block to use given "
                    "version:{} and proj_channels:{}".format(
                        settings.version, settings.proj_channels
                    )
                )

            in_channels = settings.out_channels
            stride = 1

        return Sequential(*blocks)


@ModelRegistry.register(
    key=["resnet18", "resnet_18", "resnet-18", "resnetv1_18", "resnetv1-18"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="18",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet18(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet 18 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=2, in_channels=64, out_channels=64, downsample=False
        ),
        ResNetSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, downsample=True
        ),
        ResNetSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, downsample=True
        ),
        ResNetSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, downsample=True
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnetv2_18", "resnetv2-18"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v2",
    sub_architecture="18",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnetv2_18(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet V2 18 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=2, in_channels=64, out_channels=64, downsample=False, version=2
        ),
        ResNetSectionSettings(
            num_blocks=2, in_channels=64, out_channels=128, downsample=True, version=2
        ),
        ResNetSectionSettings(
            num_blocks=2, in_channels=128, out_channels=256, downsample=True, version=2
        ),
        ResNetSectionSettings(
            num_blocks=2, in_channels=256, out_channels=512, downsample=True, version=2
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnet34", "resnet_34", "resnet-34", "resnetv1_34", "resnetv1-34"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="34",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet34(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet 34 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3, in_channels=64, out_channels=64, downsample=False
        ),
        ResNetSectionSettings(
            num_blocks=4, in_channels=64, out_channels=128, downsample=True
        ),
        ResNetSectionSettings(
            num_blocks=6, in_channels=128, out_channels=256, downsample=True
        ),
        ResNetSectionSettings(
            num_blocks=3, in_channels=256, out_channels=512, downsample=True
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnetv2_34", "resnetv2-34"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v2",
    sub_architecture="34",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnetv2_34(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet V2 34 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3, in_channels=64, out_channels=64, downsample=False, version=2
        ),
        ResNetSectionSettings(
            num_blocks=4, in_channels=64, out_channels=128, downsample=True, version=2
        ),
        ResNetSectionSettings(
            num_blocks=6, in_channels=128, out_channels=256, downsample=True, version=2
        ),
        ResNetSectionSettings(
            num_blocks=3, in_channels=256, out_channels=512, downsample=True, version=2
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnet50", "resnet_50", "resnet-50", "resnetv1_50", "resnetv1-50"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="50",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet50(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet 50 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=64,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=128,
        ),
        ResNetSectionSettings(
            num_blocks=6,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnetv2_50", "resnetv2-50"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v2",
    sub_architecture="50",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnetv2_50(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet V2 50 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=64,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=128,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=6,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
            version=2,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=[
        "resnet50_2xwidth",
        "resnet_50_2xwidth",
        "resnet-50-2xwidth",
        "resnetv1_50_2xwidth",
        "resnetv1-50-2xwidth",
    ],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="50_2x",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet50_2xwidth(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    ResNet 50 implementation where channel sizes for 3x3 convolutions are doubled;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=128,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSectionSettings(
            num_blocks=6,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=512,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=1024,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnext50", "resnext_50", "resnext-50"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnext",
    sub_architecture="50",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnext50(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNext 50 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=128,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=256,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=6,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=512,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=1024,
            groups=32,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnet101", "resnet_101", "resnet-101", "resnetv1_101", "resnetv1-101"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="101",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet101(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet 101 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=64,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=128,
        ),
        ResNetSectionSettings(
            num_blocks=23,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnetv2_101", "resnetv2-101"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v2",
    sub_architecture="101",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnetv2_101(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet V2 101 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=64,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=128,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=23,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
            version=2,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=[
        "resnet101_2xwidth",
        "resnet_101_2xwidth",
        "resnet-101-2xwidth",
        "resnetv1_101_2xwidth",
        "resnetv1-101-2xwidth",
    ],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="101_2x",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet101_2xwidth(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    ResNet 101 implementation where channel sizes for 3x3 convolutions are doubled;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=128,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSectionSettings(
            num_blocks=23,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=512,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=1024,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnext101", "resnext_101", "resnext-101"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnext",
    sub_architecture="101",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnext101(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNext 101 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=128,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=4,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=256,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=23,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=512,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=1024,
            groups=32,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnet152", "resnet_152", "resnet-152", "resnetv1_152", "resnetv1-152"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="152",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnet152(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet 152 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=64,
        ),
        ResNetSectionSettings(
            num_blocks=8,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=128,
        ),
        ResNetSectionSettings(
            num_blocks=36,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnetv2_152", "resnetv2-152"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v2",
    sub_architecture="152",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnetv2_152(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNet V2 152 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=64,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=8,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=128,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=36,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
            version=2,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
            version=2,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )


@ModelRegistry.register(
    key=["resnext152", "resnext_152", "resnext-152"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="resnext",
    sub_architecture="152",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
def resnext152(num_classes: int = 1000, class_type: str = "single") -> ResNet:
    """
    Standard ResNext 152 implementation;
    expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created ResNet Module
    """
    sec_settings = [
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=64,
            out_channels=256,
            downsample=False,
            proj_channels=128,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=8,
            in_channels=256,
            out_channels=512,
            downsample=True,
            proj_channels=256,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=36,
            in_channels=512,
            out_channels=1024,
            downsample=True,
            proj_channels=512,
            groups=32,
        ),
        ResNetSectionSettings(
            num_blocks=3,
            in_channels=1024,
            out_channels=2048,
            downsample=True,
            proj_channels=1024,
            groups=32,
        ),
    ]

    return ResNet(
        sec_settings=sec_settings, num_classes=num_classes, class_type=class_type
    )
