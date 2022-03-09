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
Simple PyTorch implementations for the MNIST dataset.
"""

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
)

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import ReLU


__all__ = ["MnistNet", "mnist_net"]


class _ConvBNRelu(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
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

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out


class _Classifier(Module):
    def __init__(self, in_channels: int, classes: int, class_type: str):
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(in_channels, classes)

        if class_type == "single":
            self.softmax = Softmax(dim=1)
        elif class_type == "multi":
            self.softmax = Sigmoid()
        else:
            raise ValueError("unknown class_type given of {}".format(class_type))

    def forward(self, inp: Tensor):
        out = self.avgpool(inp)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        classes = self.softmax(logits)

        return logits, classes


class MnistNet(Module):
    """
    A simple convolutional model created for the MNIST dataset

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    """

    def __init__(
        self,
        num_classes: int = 10,
        class_type: str = "single",
    ):
        super().__init__()
        self.blocks = Sequential(
            _ConvBNRelu(
                in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=1
            ),
            _ConvBNRelu(
                in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2
            ),
            _ConvBNRelu(
                in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1
            ),
            _ConvBNRelu(
                in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2
            ),
        )
        self.classifier = _Classifier(
            in_channels=128, classes=num_classes, class_type=class_type
        )

    def forward(self, inp: Tensor):
        out = self.blocks(inp)
        logits, classes = self.classifier(out)

        return logits, classes


@ModelRegistry.register(
    key=["mnistnet"],
    input_shape=(1, 28, 28),
    domain="cv",
    sub_domain="classification",
    architecture="mnistnet",
    sub_architecture=None,
    default_dataset="mnist",
    default_desc="base",
)
def mnist_net(num_classes: int = 10, class_type: str = "single") -> MnistNet:
    """
    MnistNet implementation; expected input shape is (B, 1, 28, 28)

    :param num_classes: the number of classes to classify
    :param class_type: one of [single, multi] to support multi class training;
        default single
    :return: The created MnistNet Module
    """
    return MnistNet(num_classes, class_type)
