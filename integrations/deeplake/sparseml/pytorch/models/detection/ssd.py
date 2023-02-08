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
Generic SSD Model framework

Information about SSD Networks can be found in the paper
`here <https://arxiv.org/abs/1512.02325>`__.
"""


from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor, nn


__all__ = [
    "SSDBackbone",
    "SSD300",
]


class SSDBackbone(ABC):
    """
    Abstract class for representing backbone models for Single Shot Detectors
    """

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        """
        :return: a list of the sizes of the addtional out channels to be used
            with this backbone
        """
        return self._out_channels

    @abstractmethod
    def get_feature_extractor(self) -> nn.Module:
        """
        :return: A feature extrator module to be used for an SSD model
        """
        pass


class _SSDHeadBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1 if downsample else 0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act_out = nn.ReLU(inplace=True)
        self.initialize()

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_out(out)

        return out

    def initialize(self):
        # convs
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        # bns
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.bias, 0.0)


class _SSDHead(nn.ModuleList):
    def __init__(self, layer_channels: List[int], hidden_channels: List[int]):
        super(_SSDHead, self).__init__()

        in_channels = layer_channels[:-1]
        out_channels = layer_channels[1:]
        assert len(in_channels) == len(out_channels) == len(hidden_channels)

        blocks = []
        for idx, (layer_in, layer_hidden, layer_out) in enumerate(
            zip(in_channels, hidden_channels, out_channels)
        ):
            downsample = idx < len(in_channels) - 2  # downsample first n-2 layers
            blocks.append(_SSDHeadBlock(layer_in, layer_hidden, layer_out, downsample))

        self.extend(blocks)

    def forward(self, inp: Tensor) -> List[Tensor]:
        outputs = [inp]
        curr_tens = inp
        for block in self:
            curr_tens = block(curr_tens)
            outputs.append(curr_tens)
        return outputs


class _SSDPredictor(nn.Module):
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
                nn.Conv2d(out_channel_size, num_boxes * 4, kernel_size=3, padding=1)
            )
            self.confidence_predictor.append(
                nn.Conv2d(
                    out_channel_size,
                    num_boxes * self._num_classes,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.initialize()

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

    def initialize(self):
        for conv in [*self.location_predictor, *self.confidence_predictor]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")


class SSD300(nn.Module):
    """
    Single Shot Detector model that takes in a generic CNN backbone

    :param backbone: SSDBackbone whose forward pass provides the feature extraction
        for an SSD model
    :param num_classes: number of target classes. Default is 81 to match
        the number of classes in the COCO detection dataset
    """

    def __init__(self, backbone: SSDBackbone, num_classes: int = 91):
        super().__init__()

        self.feature_extractor = backbone.get_feature_extractor()

        num_hidden_channels = [256, 256, 128, 128, 128]
        self.head = _SSDHead(backbone.out_channels, num_hidden_channels)

        num_default_boxes = [4, 6, 6, 6, 4, 4]
        self.predictor = _SSDPredictor(
            num_default_boxes, backbone.out_channels, num_classes
        )

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        backbone_out = self.feature_extractor(inp)
        head_outputs = self.head(backbone_out)
        locations, confidences = self.predictor(head_outputs)

        return locations, confidences
