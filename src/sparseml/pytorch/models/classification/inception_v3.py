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
PyTorch Inception V3 implementations.
Further info can be found in the paper `here <http://arxiv.org/abs/1512.00567>`__.
"""

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    ModuleList,
    Sequential,
    Sigmoid,
    Softmax,
    init,
)

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import ReLU


__all__ = ["InceptionV3", "inception_v3"]


def _init_conv_linear(mod: Conv2d, stddev: float = 0.1):
    # init as defined in the original torchvision setup
    import scipy.stats as stats

    truncnorm = stats.truncnorm(-2, 2, scale=stddev)
    values = torch.as_tensor(truncnorm.rvs(mod.weight.numel()), dtype=mod.weight.dtype)
    values = values.view(mod.weight.size())

    with torch.no_grad():
        mod.weight.copy_(values)


def _init_batch_norm(norm: BatchNorm2d, weight_const: float = 1.0):
    init.constant_(norm.weight, weight_const)
    init.constant_(norm.bias, 0.0)


class _ConvBNRelu(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        init_stddev: float = 0.1,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)
        self.act = ReLU(num_channels=out_channels, inplace=True)

        self.initialize(init_stddev)

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out

    def initialize(self, init_stddev: float):
        _init_conv_linear(self.conv, init_stddev)
        _init_batch_norm(self.bn)


class _BlockA(Module):
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        self.branch_1x1 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=64, kernel_size=1)
        )
        self.branch_3x3_3x3 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=64, kernel_size=1),
            _ConvBNRelu(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            _ConvBNRelu(in_channels=96, out_channels=96, kernel_size=3, padding=1),
        )
        self.branch_5x5 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=48, kernel_size=1),
            _ConvBNRelu(in_channels=48, out_channels=64, kernel_size=5, padding=2),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            _ConvBNRelu(
                in_channels=in_channels, out_channels=pool_features, kernel_size=1
            ),
        )

    def forward(self, x_tens: Tensor) -> Tensor:
        branch_1x1 = self.branch_1x1(x_tens)
        branch_5x5 = self.branch_5x5(x_tens)
        branch_3x3_3x3 = self.branch_3x3_3x3(x_tens)
        branch_pool = self.branch_pool(x_tens)

        out = torch.cat([branch_1x1, branch_5x5, branch_3x3_3x3, branch_pool], 1)

        return out


class _BlockB(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch_3x3 = Sequential(
            _ConvBNRelu(
                in_channels=in_channels, out_channels=384, kernel_size=3, stride=2
            )
        )
        self.branch_3x3_3x3 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=64, kernel_size=1),
            _ConvBNRelu(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            _ConvBNRelu(in_channels=96, out_channels=96, kernel_size=3, stride=2),
        )
        self.branch_pool = Sequential(MaxPool2d(kernel_size=3, stride=2))

    def forward(self, x_tens: Tensor) -> Tensor:
        branch_3x3 = self.branch_3x3(x_tens)
        branch_3x3_3x3 = self.branch_3x3_3x3(x_tens)
        branch_pool = self.branch_pool(x_tens)

        out = torch.cat([branch_3x3, branch_3x3_3x3, branch_pool], 1)

        return out


class _BlockC(Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch_1x1 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1)
        )
        self.branch_7x7 = Sequential(
            _ConvBNRelu(
                in_channels=in_channels, out_channels=channels_7x7, kernel_size=1
            ),
            _ConvBNRelu(
                in_channels=channels_7x7,
                out_channels=channels_7x7,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
            _ConvBNRelu(
                in_channels=channels_7x7,
                out_channels=192,
                kernel_size=(7, 1),
                padding=(3, 0),
            ),
        )
        self.branch_7x7_7x7 = Sequential(
            _ConvBNRelu(
                in_channels=in_channels, out_channels=channels_7x7, kernel_size=1
            ),
            _ConvBNRelu(
                in_channels=channels_7x7,
                out_channels=channels_7x7,
                kernel_size=(7, 1),
                padding=(3, 0),
            ),
            _ConvBNRelu(
                in_channels=channels_7x7,
                out_channels=channels_7x7,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
            _ConvBNRelu(
                in_channels=channels_7x7,
                out_channels=channels_7x7,
                kernel_size=(7, 1),
                padding=(3, 0),
            ),
            _ConvBNRelu(
                in_channels=channels_7x7,
                out_channels=192,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            _ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1),
        )

    def forward(self, x_tens: Tensor) -> Tensor:
        branch_1x1 = self.branch_1x1(x_tens)
        branch_7x7 = self.branch_7x7(x_tens)
        branch_7x7_7x7 = self.branch_7x7_7x7(x_tens)
        branch_pool = self.branch_pool(x_tens)

        out = torch.cat([branch_1x1, branch_7x7, branch_7x7_7x7, branch_pool], 1)

        return out


class _BlockD(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch_3x3 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1),
            _ConvBNRelu(in_channels=192, out_channels=320, kernel_size=3, stride=2),
        )
        self.branch_7x7_3x3 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1),
            _ConvBNRelu(
                in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)
            ),
            _ConvBNRelu(
                in_channels=192, out_channels=192, kernel_size=(7, 1), padding=(3, 0)
            ),
            _ConvBNRelu(in_channels=192, out_channels=192, kernel_size=3, stride=2),
        )
        self.branch_pool = Sequential(MaxPool2d(kernel_size=3, stride=2))

    def forward(self, x_tens: Tensor) -> Tensor:
        branch_3x3 = self.branch_3x3(x_tens)
        branch_7x7_3x3 = self.branch_7x7_3x3(x_tens)
        branch_pool = self.branch_pool(x_tens)

        out = torch.cat([branch_3x3, branch_7x7_3x3, branch_pool], 1)

        return out


class _BlockE(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch_1x1 = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=320, kernel_size=1)
        )
        self.branch_3x3 = ModuleList(
            [
                _ConvBNRelu(in_channels=in_channels, out_channels=384, kernel_size=1),
                _ConvBNRelu(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=(1, 3),
                    padding=(0, 1),
                ),
                _ConvBNRelu(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                ),
            ]
        )
        self.branch_3x3_3x3 = ModuleList(
            [
                _ConvBNRelu(in_channels=in_channels, out_channels=448, kernel_size=1),
                _ConvBNRelu(
                    in_channels=448, out_channels=384, kernel_size=3, padding=1
                ),
                _ConvBNRelu(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=(1, 3),
                    padding=(0, 1),
                ),
                _ConvBNRelu(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                ),
            ]
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            _ConvBNRelu(in_channels=in_channels, out_channels=192, kernel_size=1),
        )

    def forward(self, x_tens: Tensor) -> Tensor:
        branch_1x1 = self.branch_1x1(x_tens)

        branch_3x3 = self.branch_3x3[0](x_tens)
        branch_3x3 = [
            self.branch_3x3[1](branch_3x3),
            self.branch_3x3[2](branch_3x3),
        ]
        branch_3x3 = torch.cat(branch_3x3, 1)

        branch_3x3_3x3 = self.branch_3x3_3x3[0](x_tens)
        branch_3x3_3x3 = self.branch_3x3_3x3[1](branch_3x3_3x3)
        branch_3x3_3x3 = [
            self.branch_3x3_3x3[2](branch_3x3_3x3),
            self.branch_3x3_3x3[3](branch_3x3_3x3),
        ]
        branch_3x3_3x3 = torch.cat(branch_3x3_3x3, 1)

        branch_pool = self.branch_pool(x_tens)

        out = torch.cat([branch_1x1, branch_3x3, branch_3x3_3x3, branch_pool], 1)

        return out


class _Aux(Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = AvgPool2d(kernel_size=5, stride=3)
        self.adaptive_pool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.convs = Sequential(
            _ConvBNRelu(in_channels=in_channels, out_channels=128, kernel_size=1),
            _ConvBNRelu(
                in_channels=128, out_channels=768, kernel_size=5, init_stddev=0.01
            ),
        )
        self.fc = Linear(768, num_classes)

        self.initialize()

    def forward(self, x_tens: Tensor) -> Tensor:
        x_tens = self.pool(x_tens)
        x_tens = self.convs(x_tens)
        x_tens = self.adaptive_pool(x_tens)
        x_tens = torch.flatten(x_tens, 1)
        x_tens = self.fc(x_tens)

        return x_tens

    def initialize(self):
        _init_conv_linear(self.fc, stddev=0.001)


class _Classifier(Module):
    def __init__(self, in_channels: int, num_classes: int, class_type: str):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.2)
        self.fc = Linear(in_channels, num_classes)

        if class_type == "single":
            self.softmax = Softmax(dim=1)
        elif class_type == "multi":
            self.softmax = Sigmoid()
        else:
            raise ValueError("unknown class_type given of {}".format(class_type))

        self.initialize()

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.avgpool(inp)
        out = self.dropout(out)
        logits = torch.flatten(out, 1)
        logits = self.fc(logits)
        classes = self.softmax(logits)

        return logits, classes

    def initialize(self):
        _init_conv_linear(self.fc)


class InceptionV3(Module):
    """
    InceptionV3 implementation

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param enable_aux: True to enable the aux input for training,
        calculates aux logits from an earlier point in the network
        to enable smoother training as per the original paper
    """

    def __init__(self, num_classes: int, class_type: str, enable_aux: bool):
        super().__init__()
        self.section_1 = Sequential(
            _ConvBNRelu(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        )
        self.section_2 = Sequential(
            _ConvBNRelu(in_channels=32, out_channels=32, kernel_size=3),
            _ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.section_3 = Sequential(
            _ConvBNRelu(in_channels=64, out_channels=80, kernel_size=1)
        )
        self.section_4 = Sequential(
            _ConvBNRelu(in_channels=80, out_channels=192, kernel_size=3),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.section_5 = Sequential(
            _BlockA(in_channels=192, pool_features=32),
            _BlockA(in_channels=256, pool_features=64),
            _BlockA(in_channels=288, pool_features=64),
        )
        self.section_6 = Sequential(
            _BlockB(in_channels=288),
            _BlockC(in_channels=768, channels_7x7=128),
            _BlockC(in_channels=768, channels_7x7=160),
            _BlockC(in_channels=768, channels_7x7=160),
            _BlockC(in_channels=768, channels_7x7=192),
        )
        self.section_7 = Sequential(
            _BlockD(in_channels=768),
            _BlockE(in_channels=1280),
            _BlockE(in_channels=2048),
        )
        self.classifier = _Classifier(
            in_channels=2048, num_classes=num_classes, class_type=class_type
        )

        self.aux = (
            _Aux(in_channels=768, num_classes=num_classes) if enable_aux else None
        )

    def forward(self, x_tens: Tensor) -> Tuple[Tensor, ...]:
        # N x 3 x 299 x 299
        out = self.section_1(x_tens)
        # N x 32 x 149 x 149
        out = self.section_2(out)
        # N x 64 x 73 x 73
        out = self.section_3(out)
        # N x 80 x 73 x 73
        out = self.section_4(out)
        # N x 192 x 35 x 35
        out = self.section_5(out)
        # N x 288 x 35 x 35
        out = self.section_6(out)
        # N x 768 x 17 x 17
        aux_inp = out
        out = self.section_7(out)
        # N x 2048 x 8 x 8

        logits, classes = self.classifier(out)

        if self.aux is None or not self.training:
            return logits, classes

        aux = self.aux(aux_inp)

        return aux, logits, classes


@ModelRegistry.register(
    key=["inceptionv3", "inception_v3", "inception-v3"],
    input_shape=(3, 299, 299),
    domain="cv",
    sub_domain="classification",
    architecture="inception_v3",
    sub_architecture=None,
    default_dataset="imagenet",
    default_desc="base",
    def_ignore_error_tensors=[
        "classifier.fc.weight",
        "classifier.fc.bias",
        "aux.fc.weight",
        "aux.fc.bias",
    ],
)
def inception_v3(
    num_classes: int = 1000, class_type: str = "single", enable_aux: bool = True
) -> InceptionV3:
    """
    Standard InceptionV3 implementation;
    expected input shape is (B, 3, 299, 299)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :param enable_aux: True to enable the aux input for training,
        calculates aux logits from an earlier point in the network
        to enable smoother training as per the original paper
    :return: The created InceptionV3 Module
    """

    return InceptionV3(num_classes, class_type, enable_aux)
