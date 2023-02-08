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
Implementations for SSD models with ResNet backbones
"""


from typing import List, Union

from torch import nn

from sparseml.pytorch.models.detection import SSD300, SSDBackbone
from sparseml.pytorch.models.registry import ModelRegistry


__all__ = [
    "SSD300ResNetBackbone",
    "ssd300_resnet18",
    "ssd300_resnet34",
    "ssd300_resnet50",
    "ssd300_resnet101",
    "ssd300_resnet152",
]


class SSD300ResNetBackbone(SSDBackbone):
    """
    Class to provide the feature extractor and define the additional conv layers
    for an SSD300 model for various ResNet sub architecture backbones

    :param sub_arch: the ResNet sub architecture to use for this backbone
    :param pretrained: True to load pretrained ResNet weights; to load a specific
        version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path: An optional model file path to load into the created model.
        Will override pretrained parameter
    """

    def __init__(
        self,
        sub_arch: Union[str, int],
        pretrained: Union[bool, str] = True,
        pretrained_path: str = None,
    ):
        sub_arch = str(sub_arch)
        sub_architectures = ["18", "34", "50", "101", "152"]
        if sub_arch not in sub_architectures:
            raise ValueError(
                (
                    "Invalid ResNet sub architecture {}." " Valid sub architectures: {}"
                ).format(sub_arch, sub_architectures)
            )
        self._sub_arch = sub_arch
        self._pretrained = pretrained
        self._pretrained_path = pretrained_path

    @property
    def out_channels(self) -> List[int]:
        """
        :return: The number of output channels that should be used for the
            additional conv layers with this backbone
        """
        if self._sub_arch == "18":
            return [256, 512, 512, 256, 256, 128]
        elif self._sub_arch == "34":
            return [256, 512, 512, 256, 256, 256]
        else:  # "50", "101", "152"
            return [1024, 512, 512, 256, 256, 256]

    def get_feature_extractor(self) -> nn.Module:
        """
        :return: ResNet feature extrator module to be used for an SSD model
        """
        # Load ResNet model
        model_key = "resnet{}".format(self._sub_arch)
        model = ModelRegistry.create(model_key, self._pretrained, self._pretrained_path)

        input_layer, blocks, _ = model.children()
        feature_blocks = list(blocks.children())[:3]  # take first 3 ResNet blocks

        feature_extractor = nn.Sequential(input_layer, *feature_blocks)

        # set last section first block strides to 1
        last_section_first_block = feature_extractor[-1][0]
        last_section_first_block.conv1.stride = (1, 1)
        last_section_first_block.conv2.stride = (1, 1)
        last_section_first_block.identity.conv.stride = (1, 1)

        return feature_extractor


@ModelRegistry.register(
    key=["ssd300_resnet18", "ssd_resnet18"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="detection",
    architecture="ssd",
    sub_architecture="resnet18_300",
    default_dataset="coco",
    default_desc="base",
)
def ssd300_resnet18(
    num_classes: int = 91,
    pretrained_backbone: Union[bool, str] = True,
    pretrained_path_backbone: str = None,
) -> SSD300:
    """
    SSD 300 with ResNet 18 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained ResNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        created model's backbone
    :return: the created SSD ResNet model
    """
    feature_extractor = SSD300ResNetBackbone(
        "18", pretrained_backbone, pretrained_path_backbone
    )
    return SSD300(feature_extractor, num_classes)


@ModelRegistry.register(
    key=["ssd300_resnet34", "ssd_resnet34"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="detection",
    architecture="ssd",
    sub_architecture="resnet34_300",
    default_dataset="coco",
    default_desc="base",
)
def ssd300_resnet34(
    num_classes: int = 91,
    pretrained_backbone: Union[bool, str] = True,
    pretrained_path_backbone: str = None,
) -> SSD300:
    """
    SSD 300 with ResNet 34 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained ResNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        created model's backbone
    :return: the created SSD ResNet model
    """
    feature_extractor = SSD300ResNetBackbone(
        "34", pretrained_backbone, pretrained_path_backbone
    )
    return SSD300(feature_extractor, num_classes)


@ModelRegistry.register(
    key=["ssd300_resnet50", "ssd_resnet50"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="detection",
    architecture="ssd",
    sub_architecture="resnet50_300",
    default_dataset="coco",
    default_desc="base",
)
def ssd300_resnet50(
    num_classes: int = 91,
    pretrained_backbone: Union[bool, str] = True,
    pretrained_path_backbone: str = None,
) -> SSD300:
    """
    SSD 300 with ResNet 50 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained ResNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        created model's backbone
    :return: the created SSD ResNet model
    """
    feature_extractor = SSD300ResNetBackbone(
        "50", pretrained_backbone, pretrained_path_backbone
    )
    return SSD300(feature_extractor, num_classes)


@ModelRegistry.register(
    key=["ssd300_resnet101", "ssd_resnet101"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="detection",
    architecture="ssd",
    sub_architecture="resnet101_300",
    default_dataset="coco",
    default_desc="base",
)
def ssd300_resnet101(
    num_classes: int = 91,
    pretrained_backbone: Union[bool, str] = True,
    pretrained_path_backbone: str = None,
) -> SSD300:
    """
    SSD 300 with ResNet 101 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained ResNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        created model's backbone
    :return: the created SSD ResNet model
    """
    feature_extractor = SSD300ResNetBackbone(
        "101", pretrained_backbone, pretrained_path_backbone
    )
    return SSD300(feature_extractor, num_classes)


@ModelRegistry.register(
    key=["ssd300_resnet152", "ssd_resnet152"],
    input_shape=(3, 300, 300),
    domain="cv",
    sub_domain="detection",
    architecture="ssd",
    sub_architecture="resnet152_300",
    default_dataset="coco",
    default_desc="base",
)
def ssd300_resnet152(
    num_classes: int = 91,
    pretrained_backbone: Union[bool, str] = True,
    pretrained_path_backbone: str = None,
) -> SSD300:
    """
    SSD 300 with ResNet 152 backbone;
    expected input shape is (B, 3, 300, 300)

    :param num_classes: the number of classes of objects to classify
    :param pretrained_backbone: True to load pretrained ResNet weights; to load a
        specific version give a string with the name of the version (optim, optim-perf).
        Default is True
    :param pretrained_path_backbone: An optional model file path to load into the
        created model's backbone
    :return: the created SSD ResNet model
    """
    feature_extractor = SSD300ResNetBackbone(
        "152", pretrained_backbone, pretrained_path_backbone
    )
    return SSD300(feature_extractor, num_classes)
