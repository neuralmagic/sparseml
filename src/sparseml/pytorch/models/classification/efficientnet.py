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
PyTorch EfficientNet implementation
Further info can be found in the paper `here <https://arxiv.org/abs/1905.11946>`__.
"""

import math
from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Tuple

import torch
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
    SiLU,
    Softmax,
)


try:
    from torch.nn.quantized import FloatFunctional
except Exception:
    FloatFunctional = None

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import SqueezeExcite


__all__ = [
    "EfficientNet",
    "EfficientNetSectionSettings",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]


class _Add(Module):
    def __init__(self):
        super().__init__()

        if FloatFunctional:
            self.functional = FloatFunctional()
            self.wrap_qat = True
            self.qat_wrapper_kwargs = {
                "num_inputs": 2,
                "num_outputs": 0,
            }

    def forward(self, a: Tensor, b: Tensor):
        if FloatFunctional:
            return self.functional.add(a, b)
        else:
            return torch.add(a, b)


class QATSiLU(SiLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 1,
            "num_outputs": 1,
        }


class _InvertedBottleneckBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion_ratio: int,
        stride: int,
        se_ratio: Optional[float],
        se_mod: bool,
        bn_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._se_mod = se_mod
        self._bn_kwargs = bn_kwargs
        expanded_channels = _scale_num_channels(in_channels, expansion_ratio)
        squeezed_channels = (
            max(1, int(in_channels * se_ratio))
            if se_ratio and 0 < se_ratio <= 1
            else None
        )

        if bn_kwargs is None:
            bn_kwargs = {}

        self.expand = (
            Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            Conv2d(
                                in_channels=in_channels,
                                out_channels=expanded_channels,
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                        (
                            "bn",
                            BatchNorm2d(num_features=expanded_channels, **bn_kwargs),
                        ),
                        (
                            "act",
                            QATSiLU() if squeezed_channels else SiLU(),
                        ),
                    ]
                )
            )
            if expanded_channels != in_channels
            else None
        )

        spatial_padding = (kernel_size - 1) // 2
        self.spatial = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=expanded_channels,
                            out_channels=expanded_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=spatial_padding,
                            groups=expanded_channels,
                            bias=False,
                        ),
                    ),
                    ("bn", BatchNorm2d(num_features=expanded_channels, **bn_kwargs)),
                    (
                        "act",
                        QATSiLU() if squeezed_channels else SiLU(),
                    ),
                ]
            )
        )

        if self._se_mod:
            self.se = (
                SqueezeExcite(out_channels, squeezed_channels, "silu")
                if squeezed_channels
                else None
            )
        else:
            self.se = (
                SqueezeExcite(expanded_channels, squeezed_channels, "silu")
                if squeezed_channels
                else None
            )

        self.project = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=expanded_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            bias=False,
                        ),
                    ),
                    ("bn", BatchNorm2d(num_features=out_channels, **bn_kwargs)),
                ]
            )
        )

        if self._stride == 1 and self._in_channels == self._out_channels:
            self.add = _Add()
        else:
            self.add = None

    def forward(self, inp: Tensor):
        out = inp

        if self.expand is not None:
            out = self.expand(inp)

        out = self.spatial(out)

        if self.se is not None and not self._se_mod:
            out = out * self.se(out)

        out = self.project(out)

        if self.se is not None and self._se_mod:
            out = out * self.se(out)

        if self.add is not None:
            out = self.add(out, inp)

        return out


class _FusedInvertedBottleneckBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion_ratio: int,
        stride: int,
        bn_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._bn_kwargs = bn_kwargs
        expanded_channels = _scale_num_channels(in_channels, expansion_ratio)

        if bn_kwargs is None:
            bn_kwargs = {}

        if expanded_channels == in_channels:
            projection_layer = False
            expanded_channels = out_channels
        else:
            projection_layer = True

        spatial_padding = (kernel_size - 1) // 2
        self.expand = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=in_channels,
                            out_channels=expanded_channels,
                            kernel_size=kernel_size,
                            bias=False,
                            padding=spatial_padding,
                            stride=stride,
                        ),
                    ),
                    ("bn", BatchNorm2d(num_features=expanded_channels, **bn_kwargs)),
                    (
                        "act",
                        SiLU(),
                    ),
                ]
            )
        )

        if projection_layer:
            self.project = Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            Conv2d(
                                in_channels=expanded_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                        ("bn", BatchNorm2d(num_features=out_channels, **bn_kwargs)),
                    ]
                )
            )
        else:
            self.project = None

        if self._stride == 1 and self._in_channels == self._out_channels:
            self.add = _Add()
        else:
            self.add = None

    def forward(self, inp: Tensor):
        out = self.expand(inp)

        if self.project is not None:
            out = self.project(out)

        if self.add is not None:
            out = self.add(out, inp)

        return out


class _Classifier(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        classes: int,
        dropout: float,
        class_type: str,
        bn_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        if bn_kwargs is None:
            bn_kwargs = {}
        self.bn = BatchNorm2d(num_features=out_channels, **bn_kwargs)
        self.act = SiLU()
        self.pool = AdaptiveAvgPool2d(1)
        self.dropout = Dropout(p=dropout)
        self.fc = Linear(out_channels, classes)

        if class_type == "single":
            self.softmax = Softmax(dim=1)
        elif class_type == "multi":
            self.softmax = Sigmoid()
        else:
            raise ValueError("unknown class_type given of {}".format(class_type))

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes


class EfficientNetSectionSettings(object):
    """
    Settings to describe how to put together an EfficientNet architecture
    using user supplied configurations.

    :param num_blocks: the number of blocks to put in the section
    :param in_channels: the number of input channels to the section
    :param out_channels: the number of output channels from the section
    :param kernel_size: the kernel size of the depth-wise convolution
    :param expansion_ratio: (in_channels * expansion_ratio) is the number of
        input/output channels of the depth-wise convolution
    :param stride: the stride of the depth-wise convolution
    :param se_ratio: (in_channels * se_ratio) is the number of input channels
        for squeeze-excite
    :param se_mod: If true, moves squeeze-excite to the end of the block
        (after last 1x1)
    """

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion_ratio: int,
        stride: int,
        se_ratio: Optional[float] = None,
        se_mod: bool = False,
        is_fused: bool = False,
    ):
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self.se_ratio = se_ratio
        self.se_mod = se_mod
        self.is_fused = is_fused


class EfficientNet(Module):
    """
    EfficientNet implementation

    :param sec_settings: the settings for each section in the vgg model
    :param out_channels: the number of output channels in the classifier before the fc
    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    """

    def __init__(
        self,
        sec_settings: List[EfficientNetSectionSettings],
        out_channels: int,
        num_classes: int,
        class_type: str,
        dropout: float,
        bn_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()

        _bn_kwargs = bn_kwargs or {}

        self.input = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        Conv2d(
                            in_channels=3,
                            out_channels=sec_settings[0].in_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "bn",
                        BatchNorm2d(
                            num_features=sec_settings[0].in_channels, **_bn_kwargs
                        ),
                    ),
                    ("act", SiLU()),
                ]
            )
        )
        self.sections = Sequential(
            *[
                EfficientNet.create_section(settings, bn_kwargs)
                for settings in sec_settings
            ]
        )
        self.classifier = _Classifier(
            in_channels=sec_settings[-1].out_channels,
            out_channels=out_channels,
            classes=num_classes,
            dropout=dropout,
            class_type=class_type,
            bn_kwargs=bn_kwargs,
        )

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.input(inp)
        feat = self.sections(feat)
        logits, classes = self.classifier(feat)

        return logits, classes

    @staticmethod
    def create_section(
        settings: EfficientNetSectionSettings,
        bn_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Sequential:
        assert settings.num_blocks > 0

        in_channels = settings.in_channels
        stride = settings.stride
        blocks = []

        for block in range(settings.num_blocks):
            if settings.is_fused:
                blocks.append(
                    _FusedInvertedBottleneckBlock(
                        in_channels=in_channels
                        if block == 0
                        else settings.out_channels,
                        out_channels=settings.out_channels,
                        kernel_size=settings.kernel_size,
                        expansion_ratio=settings.expansion_ratio,
                        stride=stride if block == 0 else 1,
                        bn_kwargs=bn_kwargs,
                    )
                )
            else:
                blocks.append(
                    _InvertedBottleneckBlock(
                        in_channels=in_channels
                        if block == 0
                        else settings.out_channels,
                        out_channels=settings.out_channels,
                        kernel_size=settings.kernel_size,
                        expansion_ratio=settings.expansion_ratio,
                        stride=stride if block == 0 else 1,
                        se_ratio=settings.se_ratio,
                        se_mod=settings.se_mod,
                        bn_kwargs=bn_kwargs,
                    )
                )

        return Sequential(*blocks)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(math.ceil(new_v))


def _scale_num_channels(channels: int, width_mult: float) -> int:
    return _make_divisible(channels * width_mult, 8)


def _scale_num_blocks(blocks: int, depth_mult: float) -> int:
    scaled = int(math.ceil(depth_mult * blocks))

    return scaled


def _create_section_settings(
    width_mult: float, depth_mult: float, se_mod: bool
) -> Tuple[List[EfficientNetSectionSettings], int]:
    # return section settings as well as the out channels as tuple
    return (
        [
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(1, depth_mult),
                in_channels=_scale_num_channels(32, width_mult),
                out_channels=_scale_num_channels(16, width_mult),
                kernel_size=3,
                expansion_ratio=1,
                stride=1,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(2, depth_mult),
                in_channels=_scale_num_channels(16, width_mult),
                out_channels=_scale_num_channels(24, width_mult),
                kernel_size=3,
                expansion_ratio=6,
                stride=2,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(2, depth_mult),
                in_channels=_scale_num_channels(24, width_mult),
                out_channels=_scale_num_channels(40, width_mult),
                kernel_size=5,
                expansion_ratio=6,
                stride=2,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(3, depth_mult),
                in_channels=_scale_num_channels(40, width_mult),
                out_channels=_scale_num_channels(80, width_mult),
                kernel_size=3,
                expansion_ratio=6,
                stride=2,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(3, depth_mult),
                in_channels=_scale_num_channels(80, width_mult),
                out_channels=_scale_num_channels(112, width_mult),
                kernel_size=5,
                expansion_ratio=6,
                stride=1,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(4, depth_mult),
                in_channels=_scale_num_channels(112, width_mult),
                out_channels=_scale_num_channels(192, width_mult),
                kernel_size=5,
                expansion_ratio=6,
                stride=2,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
            EfficientNetSectionSettings(
                num_blocks=_scale_num_blocks(1, depth_mult),
                in_channels=_scale_num_channels(192, width_mult),
                out_channels=_scale_num_channels(320, width_mult),
                kernel_size=3,
                expansion_ratio=6,
                stride=1,
                se_ratio=0.25,
                se_mod=se_mod,
            ),
        ],
        _scale_num_channels(1280, width_mult),
    )


def _efficient_net_params(model_name):
    # Coefficients: width, depth, dropout, in_size
    params_dict = {
        "efficientnet_b0": (1.0, 1.0, 0.2, 224),
        "efficientnet_b1": (1.0, 1.1, 0.2, 240),
        "efficientnet_b2": (1.1, 1.2, 0.3, 288),
        "efficientnet_b3": (1.2, 1.4, 0.3, 300),
        "efficientnet_b4": (1.4, 1.8, 0.4, 380),
        "efficientnet_b5": (1.6, 2.2, 0.4, 456),
        "efficientnet_b6": (1.8, 2.6, 0.5, 528),
        "efficientnet_b7": (2.0, 3.1, 0.5, 600),
    }
    return params_dict[model_name]


@ModelRegistry.register(
    key=["efficientnetb0", "efficientnet_b0", "efficientnet-b0"],
    input_shape=(3, 224, 224),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b0",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b0(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.2,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B0 implementation; expected input shape is (B, 3, 224, 224)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.0
    depth_mult = 1.0
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
    )


@ModelRegistry.register(
    key=["efficientnetb1", "efficientnet_b1", "efficientnet-b1"],
    input_shape=(3, 240, 240),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b1",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b1(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.2,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B1 implementation; expected input shape is (B, 3, 240, 240)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.0
    depth_mult = 1.1
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
    )


@ModelRegistry.register(
    key=["efficientnetb2", "efficientnet_b2", "efficientnet-b2"],
    input_shape=(3, 260, 260),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b2",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b2(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.3,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B2 implementation; expected input shape is (B, 3, 260, 260)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.1
    depth_mult = 1.2
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
    )


@ModelRegistry.register(
    key=["efficientnetb3", "efficientnet_b3", "efficientnet-b3"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b3",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b3(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.3,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B3 implementation; expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.2
    depth_mult = 1.4
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
    )


@ModelRegistry.register(
    key=["efficientnetb4", "efficientnet_b4", "efficientnet-b4"],
    input_shape=(3, 380, 380),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b4",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b4(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.4,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B4 implementation; expected input shape is (B, 3, 380, 380)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.4
    depth_mult = 1.8
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
    )


@ModelRegistry.register(
    key=["efficientnetb5", "efficientnet_b5", "efficientnet-b5"],
    input_shape=(3, 456, 456),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b5",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b5(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.4,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B5 implementation; expected input shape is (B, 3, 456, 456)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.6
    depth_mult = 2.2
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
        bn_kwargs={"eps": 1.0e-03, "momentum": 0.01},
    )


@ModelRegistry.register(
    key=["efficientnetb6", "efficientnet_b6", "efficientnet-b6"],
    input_shape=(3, 528, 528),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b6",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b6(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.5,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B6 implementation; expected input shape is (B, 3, 528, 528)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 1.8
    depth_mult = 2.6
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
        bn_kwargs={"eps": 1.0e-03, "momentum": 0.01},
    )


@ModelRegistry.register(
    key=["efficientnetb7", "efficientnet_b7", "efficientnet-b7"],
    input_shape=(3, 600, 600),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet",
    sub_architecture="b7",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_b7(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.5,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNet B0 implementation; expected input shape is (B, 3, 600, 600)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :param se_mod: If true, moves squeeze-excite to the end of the block
            (after last 1x1)
    :return: The created EfficientNet B0 Module
    """

    width_mult = 2.0
    depth_mult = 3.1
    sec_settings, out_channels = _create_section_settings(
        width_mult, depth_mult, se_mod
    )

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=out_channels,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
        bn_kwargs={"eps": 1.0e-03, "momentum": 0.01},
    )


@ModelRegistry.register(
    key=[
        "efficientnet_v2_s",
        "efficientnet_v2-s",
        "efficientnetv2-s",
        "efficientnetv2_s",
    ],
    input_shape=(3, 384, 384),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet_v2",
    sub_architecture="s",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_v2_s(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.2,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNetV2-s implementation; expected input shape is (B, 3, 384, 384)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :return: The created EfficientNet_V2-S Module
    """

    sec_settings = [
        EfficientNetSectionSettings(
            num_blocks=2,
            in_channels=24,
            out_channels=24,
            kernel_size=3,
            expansion_ratio=1,
            stride=1,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=4,
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=4,
            in_channels=48,
            out_channels=64,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=6,
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=9,
            in_channels=128,
            out_channels=160,
            kernel_size=3,
            expansion_ratio=6,
            stride=1,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=15,
            in_channels=160,
            out_channels=256,
            kernel_size=3,
            expansion_ratio=6,
            stride=2,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
    ]

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=1280,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
        bn_kwargs={"eps": 1.0e-03},
    )


@ModelRegistry.register(
    key=[
        "efficientnet_v2_m",
        "efficientnet_v2-m",
        "efficientnetv2-m",
        "efficientnetv2_m",
    ],
    input_shape=(3, 480, 480),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet_v2",
    sub_architecture="m",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_v2_m(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.3,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNetV2-m implementation; expected input shape is (B, 3, 480, 480)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :return: The created EfficientNet_V2-M Module
    """

    sec_settings = [
        EfficientNetSectionSettings(
            num_blocks=3,
            in_channels=24,
            out_channels=24,
            kernel_size=3,
            expansion_ratio=1,
            stride=1,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=5,
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=5,
            in_channels=48,
            out_channels=80,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=7,
            in_channels=80,
            out_channels=160,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=14,
            in_channels=160,
            out_channels=176,
            kernel_size=3,
            expansion_ratio=6,
            stride=1,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=18,
            in_channels=176,
            out_channels=304,
            kernel_size=3,
            expansion_ratio=6,
            stride=2,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=5,
            in_channels=304,
            out_channels=512,
            kernel_size=3,
            expansion_ratio=6,
            stride=1,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
    ]

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=1280,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
        bn_kwargs={"eps": 1.0e-03},
    )


@ModelRegistry.register(
    key=[
        "efficientnet_v2_l",
        "efficientnet_v2-l",
        "efficientnetv2-l",
        "efficientnetv2_l",
    ],
    input_shape=(3, 480, 480),
    domain="cv",
    sub_domain="classification",
    architecture="efficientnet_v2",
    sub_architecture="l",
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    desc_args={"optim-perf": ("se_mod", True)},
)
def efficientnet_v2_l(
    num_classes: int = 1000,
    class_type: str = "single",
    dropout: float = 0.4,
    se_mod: bool = False,
) -> EfficientNet:
    """
    EfficientNetV2-l implementation; expected input shape is (B, 3, 480, 480)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param dropout: the amount of dropout to use while training
    :return: The created EfficientNet_V2-L Module
    """

    sec_settings = [
        EfficientNetSectionSettings(
            num_blocks=4,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            expansion_ratio=1,
            stride=1,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=7,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=7,
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            is_fused=True,
        ),
        EfficientNetSectionSettings(
            num_blocks=10,
            in_channels=96,
            out_channels=192,
            kernel_size=3,
            expansion_ratio=4,
            stride=2,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=19,
            in_channels=192,
            out_channels=224,
            kernel_size=3,
            expansion_ratio=6,
            stride=1,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=25,
            in_channels=224,
            out_channels=384,
            kernel_size=3,
            expansion_ratio=6,
            stride=2,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
        EfficientNetSectionSettings(
            num_blocks=7,
            in_channels=384,
            out_channels=640,
            kernel_size=3,
            expansion_ratio=6,
            stride=1,
            se_ratio=0.25,
            se_mod=se_mod,
        ),
    ]

    return EfficientNet(
        sec_settings=sec_settings,
        out_channels=1280,
        num_classes=num_classes,
        class_type=class_type,
        dropout=dropout,
        bn_kwargs={"eps": 1.0e-03},
    )
