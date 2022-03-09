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
Keras ResNet implementation.
Further info on ResNet can be found in the paper
`here <https://arxiv.org/abs/1512.03385>`__.
"""

from typing import List, Union

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from sparseml.keras.models.registry import ModelRegistry
from sparseml.keras.utils import keras


__all__ = ["ResNetSection", "resnet_const", "resnet50", "resnet101", "resnet152"]


BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5

BASE_NAME_SCOPE = "resnet"


def _expand_name(prefix: str, suffix: str, sep: str = "."):
    return prefix + sep + suffix


def _input(
    name: str,
    x_tens: tensorflow.Tensor,
    training: Union[bool, tensorflow.Tensor],
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tensorflow.Tensor:
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=_expand_name(name, "pad"))(
        x_tens
    )
    x = layers.Conv2D(
        64, 7, strides=2, use_bias=False, name=_expand_name(name, "conv")
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=BN_EPSILON, name=_expand_name(name, "bn")
    )(x)
    x = layers.Activation("relu", name=_expand_name(name, "relu"))(x)
    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=_expand_name(name, "pool_pad")
    )(x)
    x = layers.MaxPooling2D(3, strides=2, name=_expand_name(name, "pool_pool"))(x)

    return x


def _identity_modifier(
    name: str,
    x_tens: tensorflow.Tensor,
    training: Union[bool, tensorflow.Tensor],
    out_channels: int,
    stride: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tensorflow.Tensor:
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    shortcut = layers.Conv2D(
        out_channels, 1, strides=stride, name=_expand_name(name, "conv")
    )(x_tens)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=BN_EPSILON, name=_expand_name(name, "bn")
    )(shortcut)
    return shortcut


def _bottleneck_block(
    name: str,
    x_tens: tensorflow.Tensor,
    training: Union[bool, tensorflow.Tensor],
    out_channels: int,
    proj_channels: int,
    stride: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tensorflow.Tensor:
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    x = layers.Conv2D(proj_channels, 1, name=_expand_name(name, "conv1"))(x_tens)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=BN_EPSILON, name=_expand_name(name, "bn1")
    )(x)
    x = layers.Activation("relu", name=_expand_name(name, "relu1"))(x)

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=_expand_name(name, "pad_conv2")
    )(x)
    x = layers.Conv2D(
        proj_channels, 3, strides=stride, name=_expand_name(name, "conv2")
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=BN_EPSILON, name=_expand_name(name, "bn2")
    )(x)
    x = layers.Activation("relu", name=_expand_name(name, "relu2"))(x)

    x = layers.Conv2D(out_channels, 1, name=_expand_name(name, "conv3"))(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=BN_EPSILON, name=_expand_name(name, "bn3")
    )(x)

    if stride > 1 or int(x_tens.shape[3]) != out_channels:
        shortcut = _identity_modifier(
            _expand_name(name, "identity"),
            x_tens,
            training,
            out_channels,
            stride,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
    else:
        shortcut = x_tens

    x = layers.Add(name=_expand_name(name, "add"))([shortcut, x])
    x = layers.Activation("relu", name=_expand_name(name, "out"))(x)

    return x


def _classifier(
    name: str,
    x_tens: tensorflow.Tensor,
    training: Union[bool, tensorflow.Tensor],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tensorflow.Tensor:
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x_tens)
    if num_classes:
        if class_type:
            if class_type == "single":
                act = "softmax"
            elif class_type == "multi":
                act = "sigmoid"
            else:
                raise ValueError("unknown class_type given of {}".format(class_type))
        else:
            act = None

        outputs = layers.Dense(
            num_classes, activation=act, name=_expand_name(name, "fc")
        )(x)
    else:
        outputs = x
    return outputs


class ResNetSection(object):
    """
    Settings to describe how to put together a ResNet based architecture
    using user supplied configurations.

    :param num_blocks: the number of blocks to put in the section
        (ie Basic or Bottleneck blocks)
    :param out_channels: the number of output channels from the section
    :param downsample: True to apply stride 2 for downsampling of the input,
        False otherwise
    :param proj_channels: The number of channels in the projection for a
        bottleneck block, if < 0 then uses basic
    """

    def __init__(
        self,
        num_blocks: int,
        out_channels: int,
        downsample: bool,
        proj_channels: int = -1,
    ):
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.downsample = downsample
        self.proj_channels = proj_channels

    def create(
        self,
        name: str,
        x_tens: tensorflow.Tensor,
        training: Union[bool, tensorflow.Tensor],
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    ) -> tensorflow.Tensor:
        """
        Create the section in the current graph and scope

        :param name: the name for the scope to create the section under
        :param x_tens: The input tensor to the ResNet architecture
        :param training: bool or Tensor to specify if the model should be run
            in training or inference mode
        :param kernel_initializer: Initializer to use for the conv and
            fully connected kernels
        :param bias_initializer: Initializer to use for the bias in the fully connected
        :param beta_initializer: Initializer to use for the batch norm beta variables
        :param gamma_initializer: Initializer to use for the batch norm gama variables
        :return: the output tensor from the section
        """
        out = x_tens

        stride = 2 if self.downsample else 1

        for block in range(self.num_blocks):
            block_name = _expand_name(name, "{}".format(block))
            if self.proj_channels > 0:
                out = _bottleneck_block(
                    name=block_name,
                    x_tens=out,
                    training=training,
                    out_channels=self.out_channels,
                    proj_channels=self.proj_channels,
                    stride=stride,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    beta_initializer=beta_initializer,
                    gamma_initializer=gamma_initializer,
                )

            stride = 1

        return out


def resnet_const(
    x_tens: tensorflow.Tensor,
    training: Union[bool, tensorflow.Tensor],
    sec_settings: List[ResNetSection],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> keras.models.Model:
    """
    Graph constructor for ResNet implementation.

    :param x_tens: The input tensor to the ResNet architecture
    :param training: bool or Tensor to specify if the model should be run
        in training or inference mode
    :param sec_settings: The settings for each section in the ResNet modoel
    :param num_classes: The number of classes to classify
    :param class_type: One of [single, multi, None] to support multi class training.
        Default single. If None, then will not add the fully connected at the end.
    :param kernel_initializer: Initializer to use for the conv and
        fully connected kernels
    :param bias_initializer: Initializer to use for the bias in the fully connected
    :param beta_initializer: Initializer to use for the batch norm beta variables
    :param gamma_initializer: Initializer to use for the batch norm gama variables
    :return: the output tensor from the created graph
    """
    channels_last = K.image_data_format() == "channels_last"
    if x_tens is None:
        input_shape = (224, 224, 3) if channels_last else (3, 224, 224)
        x_tens = layers.Input(shape=input_shape)

    out = _input(
        "input",
        x_tens,
        training,
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    )

    for sec_index, section in enumerate(sec_settings):
        out = section.create(
            name="sections.{}".format(sec_index),
            x_tens=out,
            training=training,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )

    outputs = _classifier(
        "classifier",
        out,
        training,
        num_classes,
        class_type,
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    )

    return Model(inputs=x_tens, outputs=outputs)


@ModelRegistry.register(
    key=["resnet50", "resnet_50", "resnet-50", "resnetv1_50", "resnetv1-50"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="50",
    default_dataset="imagenet",
    default_desc="base",
)
def resnet50(
    inputs: tensorflow.Tensor = None,
    training: Union[bool, tensorflow.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=keras.initializers.GlorotUniform(),
    bias_initializer=keras.initializers.GlorotUniform(),
    beta_initializer=keras.initializers.GlorotUniform(),
    gamma_initializer=keras.initializers.GlorotUniform(),
) -> keras.models.Model:
    """
    Standard ResNet50 implementation;
    expected input shape is (B, 224, 224, 3)

    :param inputs: The input tensor to the ResNet architecture
    :param training: bool or Tensor to specify if the model should be run
        in training or inference mode
    :param num_classes: The number of classes to classify
    :param class_type: One of [single, multi, None] to support multi class training.
        Default single. If None, then will not add the fully connected at the end.
    :param kernel_initializer: Initializer to use for the conv and
        fully connected kernels
    :param bias_initializer: Initializer to use for the bias in the fully connected
    :param beta_initializer: Initializer to use for the batch norm beta variables
    :param gamma_initializer: Initializer to use for the batch norm gama variables
    :return: the output tensor from the created graph
    """
    sec_settings = [
        ResNetSection(
            num_blocks=3,
            out_channels=256,
            downsample=False,
            proj_channels=64,
        ),
        ResNetSection(
            num_blocks=4,
            out_channels=512,
            downsample=True,
            proj_channels=128,
        ),
        ResNetSection(
            num_blocks=6,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSection(
            num_blocks=3,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
        ),
    ]

    return resnet_const(
        inputs,
        training,
        sec_settings,
        num_classes,
        class_type,
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    )


@ModelRegistry.register(
    key=["resnet101", "resnet_101", "resnet-101", "resnetv1_101", "resnetv1-101"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="101",
    default_dataset="imagenet",
    default_desc="base",
)
def resnet101(
    inputs: tensorflow.Tensor = None,
    training: Union[bool, tensorflow.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=keras.initializers.GlorotUniform(),
    bias_initializer=keras.initializers.GlorotUniform(),
    beta_initializer=keras.initializers.GlorotUniform(),
    gamma_initializer=keras.initializers.GlorotUniform(),
) -> keras.models.Model:
    """
    Standard ResNet101 implementation;
    expected input shape is (B, 224, 224, 3)

    :param inputs: The input tensor to the ResNet architecture
    :param training: bool or Tensor to specify if the model should be run
        in training or inference mode
    :param num_classes: The number of classes to classify
    :param class_type: One of [single, multi, None] to support multi class training.
        Default single. If None, then will not add the fully connected at the end.
    :param kernel_initializer: Initializer to use for the conv and
        fully connected kernels
    :param bias_initializer: Initializer to use for the bias in the fully connected
    :param beta_initializer: Initializer to use for the batch norm beta variables
    :param gamma_initializer: Initializer to use for the batch norm gama variables
    :return: the output tensor from the created graph
    """
    sec_settings = [
        ResNetSection(
            num_blocks=3,
            out_channels=256,
            downsample=False,
            proj_channels=64,
        ),
        ResNetSection(
            num_blocks=4,
            out_channels=512,
            downsample=True,
            proj_channels=128,
        ),
        ResNetSection(
            num_blocks=23,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSection(
            num_blocks=3,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
        ),
    ]

    return resnet_const(
        inputs,
        training,
        sec_settings,
        num_classes,
        class_type,
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    )


@ModelRegistry.register(
    key=["resnet152", "resnet_152", "resnet-152", "resnetv1_152", "resnetv1-152"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="152",
    default_dataset="imagenet",
    default_desc="base",
)
def resnet152(
    inputs: tensorflow.Tensor = None,
    training: Union[bool, tensorflow.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=keras.initializers.GlorotUniform(),
    bias_initializer=keras.initializers.GlorotUniform(),
    beta_initializer=keras.initializers.GlorotUniform(),
    gamma_initializer=keras.initializers.GlorotUniform(),
) -> keras.models.Model:
    """
    Standard ResNet152 implementation;
    expected input shape is (B, 224, 224, 3)

    :param inputs: The input tensor to the ResNet architecture
    :param training: bool or Tensor to specify if the model should be run
        in training or inference mode
    :param num_classes: The number of classes to classify
    :param class_type: One of [single, multi, None] to support multi class training.
        Default single. If None, then will not add the fully connected at the end.
    :param kernel_initializer: Initializer to use for the conv and
        fully connected kernels
    :param bias_initializer: Initializer to use for the bias in the fully connected
    :param beta_initializer: Initializer to use for the batch norm beta variables
    :param gamma_initializer: Initializer to use for the batch norm gama variables
    :return: the output tensor from the created graph
    """
    sec_settings = [
        ResNetSection(
            num_blocks=3,
            out_channels=256,
            downsample=False,
            proj_channels=64,
        ),
        ResNetSection(
            num_blocks=8,
            out_channels=512,
            downsample=True,
            proj_channels=128,
        ),
        ResNetSection(
            num_blocks=36,
            out_channels=1024,
            downsample=True,
            proj_channels=256,
        ),
        ResNetSection(
            num_blocks=3,
            out_channels=2048,
            downsample=True,
            proj_channels=512,
        ),
    ]

    return resnet_const(
        inputs,
        training,
        sec_settings,
        num_classes,
        class_type,
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    )
