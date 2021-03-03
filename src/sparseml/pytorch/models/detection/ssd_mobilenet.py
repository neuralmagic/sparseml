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
Implementations for SSD models with MobileNet backbones
"""


from typing import List, Union

from torch import nn

from sparseml.pytorch.models.detection import SSD300Lite, SSDBackbone
from sparseml.pytorch.models.registry import ModelRegistry


__all__ = [
    "SSD300MobileNetBackbone",
    "ssd300lite_mobilenetv2",
]


class SSD300MobileNetBackbone(SSDBackbone):
    """
    Class to provide the feature extractor and define the additional conv layers
    for an SSD300 model for various MobileNet architecture backbones

    :param version: the MobileNet version to use for this backbone
    :param pretrained: True to load pretrained MobileNet weights; to load a specific
        version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path: An optional model file path to load into the created model.
        Will override pretrained parameter
    """

    def __init__(
        self,
        version: Union[str, int] = "2",
        pretrained: Union[bool, str] = True,
        pretrained_path: str = None,
    ):
        version = int(version)
        assert int(version) in [1, 2]

        self._version = version
        self._pretrained = pretrained
        self._pretrained_path = pretrained_path

    @property
    def out_channels(self) -> List[int]:
        """
        :return: The number of output channels that should be used for the
            additional conv layers with this backbone
        """
        if self._version == 1:
            return [1024, 512, 512, 256, 256, 256]
        else:
            return [(96, 320), 512, 512, 256, 256, 256]

    def get_feature_extractor(self) -> nn.Module:
        """
        :return: MobileNet feature extractor module to be used for an SSD model
        """
        # Load MobileNet model
        model_key = "mobilenet-v{}".format(self._version)
        model = ModelRegistry.create(model_key, self._pretrained, self._pretrained_path)

        # increase feature map to 38x38
        if self._version == 2:
            model.sections[3][0].spatial.conv.stride = (1, 1)
            model.sections[5][0].spatial.conv.stride = (1, 1)

        feature_blocks = list(model.sections.children())

        return nn.Sequential(*feature_blocks)


@ModelRegistry.register(
    key=["ssd300lite_mobilenetv2", "ssdlite_mobilenetv2"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="detection",
    architecture="ssd_lite",
    sub_architecture="mobilenet_v2",
    default_dataset="coco",
    default_desc="base",
)
def ssd300lite_mobilenetv2(
    num_classes: int = 91,
    pretrained_backbone: Union[bool, str] = True,
    pretrained_path_backbone: str = None,
) -> SSD300Lite:
    """
    SSD 300 Lite with MobileNet V2 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained MobileNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        created model's backbone
    :return: the created SSD Lite MobileNet model
    """
    feature_extractor = SSD300MobileNetBackbone(
        "2", pretrained_backbone, pretrained_path_backbone
    )
    return SSD300Lite(feature_extractor, 4, num_classes)
