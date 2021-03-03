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
Generic SSD Lite Model framework

Information about SSD Lite and mobilenet v2 can be found in the paper
`here <https://arxiv.org/pdf/1801.04381>`__.
"""

from typing import List, Tuple

import torch
from torch import Tensor, nn

from sparseml.pytorch.models.detection.ssd import SSDBackbone


__all__ = [
    "SSD300Lite",
]


class _ConvBNRelu(nn.Module):
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
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)

        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)


class _SSDLiteHeadBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.conv = _ConvBNRelu(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.depth = _ConvBNRelu(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1 if downsample else 0,
            groups=hidden_channels,
        )
        self.point = _ConvBNRelu(
            hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv(inp)
        out = self.depth(out)
        out = self.point(out)

        return out


class _SSDLiteHead(nn.ModuleList):
    def __init__(self, layer_channels: List[int], hidden_channels: List[int]):
        super(_SSDLiteHead, self).__init__()

        in_channels = layer_channels[:-1]
        out_channels = layer_channels[1:]
        assert len(in_channels) == len(out_channels) == len(hidden_channels)

        blocks = []
        for idx, (layer_in, layer_hidden, layer_out) in enumerate(
            zip(in_channels, hidden_channels, out_channels)
        ):
            downsample = idx < len(in_channels) - 2  # downsample first n-2 layers
            blocks.append(
                _SSDLiteHeadBlock(layer_in, layer_hidden, layer_out, downsample)
            )

        self.extend(blocks)

    def forward(self, inp: Tensor) -> List[Tensor]:
        outputs = []
        curr_tens = inp
        for block in self:
            curr_tens = block(curr_tens)
            outputs.append(curr_tens)
        return outputs


class _SSDLitePredictorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.depth = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.point = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=1)
        self.initialize()

    def forward(self, inp: Tensor):
        out = self.depth(inp)
        out = self.point(out)

        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.depth.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.point.weight, mode="fan_out", nonlinearity="relu")


class _SSDLitePredictor(nn.Module):
    def __init__(
        self, num_default_boxes: List[int], in_channels: List[int], num_classes: int
    ):
        super().__init__()
        assert len(num_default_boxes) == len(in_channels)

        self._num_classes = num_classes

        self.location_predictor = nn.ModuleList()
        self.confidence_predictor = nn.ModuleList()
        for num_boxes, out_channel_size in zip(num_default_boxes, in_channels):
            self.location_predictor.append(
                _SSDLitePredictorBlock(out_channel_size, num_boxes * 4)
            )
            self.confidence_predictor.append(
                _SSDLitePredictorBlock(out_channel_size, num_boxes * self._num_classes)
            )

    def forward(self, ssd_head_outputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        assert len(ssd_head_outputs) == len(self.location_predictor)

        locations = []
        confidences = []
        batch_size = ssd_head_outputs[0].size(0)

        for idx, block_output in enumerate(ssd_head_outputs):
            locations.append(
                self.location_predictor[idx](block_output).view(batch_size, 4, -1)
            )
            confidences.append(
                self.confidence_predictor[idx](block_output).view(
                    batch_size, self._num_classes, -1
                )
            )

        # flatten along box dimension
        locations = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()

        # shapes: batch_size, {4, num_classes}, total_num_default_boxes
        return locations, confidences


class SSD300Lite(nn.Module):
    """
    Single Shot Detector model that takes in a CNN backbone and uses depthwise
    convolutions to perform fast object detection.

    :param backbone: SSDBackbone whose forward pass provides the feature extraction
        for an SSD model
    :param backbone_early_output_idx: index of backbone Sequential object to be used
        as input to the first layer of the SSD head.  All other layers use the entire
        backbone as input as per the SSDLite paper
    :param num_classes: number of target classes. Default is 81 to match
        the number of classes in the COCO detection dataset
    """

    def __init__(
        self,
        backbone: SSDBackbone,
        backbone_early_output_idx: int,
        num_classes: int = 91,
    ):
        super().__init__()

        self.feature_extractor = backbone.get_feature_extractor()
        self._backbone_early_output_idx = backbone_early_output_idx

        # get out channels for both backbone outputs used for this model
        backbone_output_channels = backbone.out_channels[0]
        head_channels = backbone.out_channels
        head_channels[0] = backbone_output_channels[1]  # use full backbone output
        num_hidden_channels = [256, 256, 128, 128, 128]
        self.head = _SSDLiteHead(head_channels, num_hidden_channels)

        num_default_boxes = [4, 6, 6, 6, 4, 4]
        pred_channels = backbone.out_channels
        pred_channels[0] = backbone_output_channels[0]  # use skip-layer backbone output
        self.predictor = _SSDLitePredictor(
            num_default_boxes, pred_channels, num_classes
        )

    def forward(self, tens: Tensor) -> Tuple[Tensor, Tensor]:
        for idx, block in enumerate(self.feature_extractor):
            tens = block(tens)
            if idx == self._backbone_early_output_idx:
                first_output = tens
        head_outputs = [first_output] + self.head(tens)
        locations, confidences = self.predictor(head_outputs)

        return locations, confidences
