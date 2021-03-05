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
PyTorch YoloV3 implementation.
"""


import math
from typing import List, Tuple, Union

from torch import Tensor, cat
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    Module,
    ModuleList,
    Parameter,
    Upsample,
    init,
)

from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.nn import Hardswish


__all__ = [
    "YoloV3",
    "yolo_v3",
]


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")


def _init_batch_norm(norm: BatchNorm2d, weight_const: float = 1.0):
    init.constant_(norm.weight, weight_const)
    init.constant_(norm.bias, 0.0)


class _ConvBnBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels, momentum=0.03, eps=1e-4)
        self.act = Hardswish(num_channels=out_channels, inplace=True)

        self.initialize()

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out

    def initialize(self):
        _init_conv(self.conv)
        _init_batch_norm(self.bn)


class _SPP(Module):
    def __init__(self):
        super().__init__()
        self.pool1 = MaxPool2d(kernel_size=5, stride=1, padding=2)  # pad = (k - 1) // 2
        self.pool2 = MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, inp: Tensor) -> Tensor:
        outputs = list()
        outputs.append(inp)
        outputs.append(self.pool1(inp))
        outputs.append(self.pool2(inp))
        outputs.append(self.pool3(inp))
        return cat(outputs, 1)


class _YoloConvsBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_k1: int,
        out_channels_k3: int,
        total_layers: int,
        spp_layer_idxs: List[int] = None,
    ):
        super().__init__()

        # alterante between 1x1 and 3x3 convs
        # replace layers in spp_layer_idxs with SPP layers
        layers = []
        spp_layer_idxs = spp_layer_idxs or []
        for i in range(total_layers):
            if i not in spp_layer_idxs:
                kernel_size = 1 if i % 2 == 0 else 3
                out_channels = out_channels_k1 if kernel_size == 1 else out_channels_k3
                layers.append(_ConvBnBlock(in_channels, out_channels, kernel_size))
            else:
                out_channels = in_channels * 4
                layers.append(_SPP())
            in_channels = out_channels
        self.layers = ModuleList(layers)

    def forward(self, tens: Tensor) -> Tuple[Tensor, Tensor]:
        # Return output of all layers in sequence and second to last layer for routing
        for block in self.layers[:-1]:
            tens = block(tens)
        return self.layers[-1](tens), tens


class _UpsampleBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()

        self.conv = _ConvBnBlock(in_channels, out_channels, kernel_size=1)
        self.upsample = Upsample(scale_factor=scale_factor)

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv(inp)
        out = self.upsample(out)

        return out


class _YoloDetectionBlock(Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs_per_anchor = 5 + num_classes  # 4 bbox coords + 1 obj. score

        conv_out_channels = self.num_anchors * self.num_outputs_per_anchor
        self.conv = Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.initialize()

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv(inp)
        # reshape to (bs, anchor, feature, feature, outputs)
        bs, _, feat_y, feat_x = out.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        out = (
            out.view(bs, self.num_anchors, self.num_outputs_per_anchor, feat_y, feat_x)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        return out

    def initialize(self):
        _init_conv(self.conv)

        # smart bias initialization
        b = self.conv.bias.view(3, -1).detach()
        b[:, 4] += math.log(8 / 640 ** 2)  # 8 objects per 640 image
        b[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
        self.conv.bias = Parameter(b.view(-1), requires_grad=True)


class YoloV3(Module):
    """
    Yolo v3 implementation matching standard Yolo v3 SPP configuration

    :param num_classes: the number of classes to classify objects with
    :param backbone: CNN backbone to this model
    :param backbone_out_channels: The number of output channels in each of the
        backbone's outputs.
    :param anchor_groups: List of 3x2 Tensors of anchor point coordinates for
        each of this model's detectors
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Module,
        backbone_out_channels: List[int],
        anchor_groups: List[Tensor],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        assert len(backbone_out_channels) == 3  # only 3 supported for now
        self.head = ModuleList(
            [
                _YoloConvsBlock(
                    in_channels=backbone_out_channels[0],
                    out_channels_k1=512,
                    out_channels_k3=1024,
                    total_layers=8,
                    spp_layer_idxs=[3],  # replace third conv with SPP
                ),
                _UpsampleBlock(in_channels=512, out_channels=256, scale_factor=2),
                _YoloConvsBlock(
                    in_channels=256 + backbone_out_channels[1],
                    out_channels_k1=256,
                    out_channels_k3=512,
                    total_layers=6,
                ),
                _UpsampleBlock(in_channels=256, out_channels=128, scale_factor=2),
                _YoloConvsBlock(
                    in_channels=128 + backbone_out_channels[2],
                    out_channels_k1=128,
                    out_channels_k3=256,
                    total_layers=6,
                ),
            ]
        )
        self.detector = ModuleList(
            [
                _YoloDetectionBlock(1024, num_classes, 3),
                _YoloDetectionBlock(512, num_classes, 3),
                _YoloDetectionBlock(256, num_classes, 3),
            ]
        )
        self.anchor_groups = anchor_groups

    def forward(self, inp: Tensor):
        backbone_features = self.backbone(inp)

        # run yolo head
        backbone_feature_idx = 0  # track backbone output to use
        current_tens = None  # holds the latest output to be propagated through model
        detector_inputs = []

        for section in self.head:
            if isinstance(section, _YoloConvsBlock):
                section_input = (
                    backbone_features[backbone_feature_idx]
                    if current_tens is None
                    else cat([current_tens, backbone_features[backbone_feature_idx]], 1)
                )
                detector_input, current_tens = section(section_input)
                detector_inputs.append(detector_input)
                backbone_feature_idx += 1
            else:
                current_tens = section(current_tens)

        # run detector layers
        outputs = []
        for detector_input, detector_layer in zip(detector_inputs, self.detector):
            outputs.append(detector_layer(detector_input))

        return outputs


@ModelRegistry.register(
    key=["yolo", "yolo-v3", "yolo_v3"],
    input_shape=(3, 640, 640),
    domain="cv",
    sub_domain="detection",
    architecture="yolo_v3",
    sub_architecture="spp",
    default_dataset="coco",
    default_desc="base",
)
def yolo_v3(
    num_classes: int = 80,
    pretrained_backbone: Union[bool, str] = False,
    pretrained_path_backbone: str = None,
) -> YoloV3:
    """
    Yolo-V3 model with standard DarkNet-53 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained DarkNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        DarkNet backbone. Default is None
    :return: the created Yolo model
    """
    backbone = ModelRegistry.create(
        "darknet53", pretrained_backbone, pretrained_path_backbone
    )
    backbone.as_yolo_backbone([4, 3, 2])  # set outputs to last 3 residual layers
    del backbone.classifier  # remove fc layer from state dict
    backbone_out_channels = [1024, 512, 256]
    anchor_groups = [
        Tensor([[116, 90], [156, 198], [373, 326]]),
        Tensor([[30, 61], [62, 45], [59, 119]]),
        Tensor([[10, 13], [16, 30], [33, 23]]),
    ]
    return YoloV3(num_classes, backbone, backbone_out_channels, anchor_groups)
