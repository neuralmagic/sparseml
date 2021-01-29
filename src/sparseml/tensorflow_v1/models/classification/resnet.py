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
TensorFlow ResNet implementation.
Further info on ResNet can be found in the paper
`here <https://arxiv.org/abs/1512.03385>`__.
"""

from typing import List, Union

from sparseml.tensorflow_v1.models.estimator import ClassificationEstimatorModelFn
from sparseml.tensorflow_v1.models.registry import ModelRegistry
from sparseml.tensorflow_v1.nn import activation, conv2d_block, dense_block, pool2d
from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "ResNetSection",
    "resnet_const",
    "resnet18",
    "resnet20",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


BASE_NAME_SCOPE = "resnet"


def _input(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
    simplified_arch: bool = False,
) -> tf_compat.Tensor:
    if not simplified_arch:
        out = conv2d_block(
            "input",
            x_tens,
            training,
            channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = pool2d(
            name="pool", x_tens=out, type_="max", pool_size=3, strides=2, padding=1
        )
    else:
        out = conv2d_block(
            "input",
            x_tens,
            training,
            channels=16,
            kernel_size=3,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
    return out


def _identity_modifier(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    out_channels: int,
    stride: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    out = conv2d_block(
        "identity",
        x_tens,
        training,
        out_channels,
        kernel_size=1,
        stride=stride,
        act=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
    )

    return out


def _basic_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    out_channels: int,
    stride: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = conv2d_block(
            "conv_bn_0",
            x_tens,
            training,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = conv2d_block(
            "conv_bn_1",
            out,
            training,
            out_channels,
            kernel_size=3,
            padding=1,
            act=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )

        if stride > 1 or int(x_tens.shape[3]) != out_channels:
            out = tf_compat.add(
                out,
                _identity_modifier(
                    x_tens,
                    training,
                    out_channels,
                    stride,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    beta_initializer=beta_initializer,
                    gamma_initializer=gamma_initializer,
                ),
            )
        else:
            out = tf_compat.add(out, x_tens)

        out = activation(out, act="relu", name="act_out")

    return out


def _bottleneck_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    out_channels: int,
    proj_channels: int,
    stride: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = conv2d_block(
            "conv_bn_0",
            x_tens,
            training,
            proj_channels,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = conv2d_block(
            "conv_bn_1",
            out,
            training,
            proj_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = conv2d_block(
            "conv_bn_2",
            out,
            training,
            out_channels,
            kernel_size=1,
            act=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )

        if stride > 1 or int(x_tens.shape[3]) != out_channels:
            out = tf_compat.add(
                out,
                _identity_modifier(
                    x_tens,
                    training,
                    out_channels,
                    stride,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    beta_initializer=beta_initializer,
                    gamma_initializer=gamma_initializer,
                ),
            )
        else:
            out = tf_compat.add(out, x_tens)

        out = activation(out, act="relu", name="act_out")

    return out


def _classifier(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    with tf_compat.variable_scope("classifier", reuse=tf_compat.AUTO_REUSE):
        logits = pool2d(name="avgpool", x_tens=x_tens, type_="global_avg", pool_size=1)

        if num_classes:
            logits = tf_compat.reshape(logits, [-1, int(logits.shape[3])])

            if class_type:
                if class_type == "single":
                    act = "softmax"
                elif class_type == "multi":
                    act = "sigmoid"
                else:
                    raise ValueError(
                        "unknown class_type given of {}".format(class_type)
                    )
            else:
                act = None

            logits = dense_block(
                "dense",
                logits,
                training,
                num_classes,
                include_bn=False,
                act=act,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
            )

    return logits


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
        x_tens: tf_compat.Tensor,
        training: Union[bool, tf_compat.Tensor],
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    ) -> tf_compat.Tensor:
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

        with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
            stride = 2 if self.downsample else 1

            for block in range(self.num_blocks):
                if self.proj_channels > 0:
                    out = _bottleneck_block(
                        name="block_{}".format(block),
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
                else:
                    out = _basic_block(
                        name="block_{}".format(block),
                        x_tens=out,
                        training=training,
                        out_channels=self.out_channels,
                        stride=stride,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        beta_initializer=beta_initializer,
                        gamma_initializer=gamma_initializer,
                    )

                stride = 1

        return out


def resnet_const(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    sec_settings: List[ResNetSection],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
    simplified_arch: bool = False,
) -> tf_compat.Tensor:
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
    :param simplified_arch: Whether the network is a simplified version for the
        Cifar10/100 dataset
    :return: the output tensor from the created graph
    """
    with tf_compat.variable_scope(BASE_NAME_SCOPE, reuse=tf_compat.AUTO_REUSE):
        out = _input(
            x_tens,
            training,
            kernel_initializer,
            bias_initializer,
            beta_initializer,
            gamma_initializer,
            simplified_arch=simplified_arch,
        )

        for sec_index, section in enumerate(sec_settings):
            out = section.create(
                name="section_{}".format(sec_index),
                x_tens=out,
                training=training,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
            )

        logits = _classifier(
            out,
            training,
            num_classes,
            class_type,
            kernel_initializer,
            bias_initializer,
            beta_initializer,
            gamma_initializer,
        )

    return logits


@ModelRegistry.register(
    key=["resnet18", "resnet_18", "resnet-18", "resnetv1_18", "resnetv1-18"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="18",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def resnet18(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:
    """
    Standard ResNet18 implementation;
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
        ResNetSection(num_blocks=2, out_channels=64, downsample=False),
        ResNetSection(num_blocks=2, out_channels=128, downsample=True),
        ResNetSection(num_blocks=2, out_channels=256, downsample=True),
        ResNetSection(num_blocks=2, out_channels=512, downsample=True),
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
    key=["resnet20", "resnet_20", "resnet-20", "resnetv1_20", "resnetv1-20"],
    input_shape=(32, 32, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="20",
    default_dataset="Cifar10",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def resnet20(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 10,
    class_type: str = "single",
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:

    with tf_compat.variable_scope("resnet20", reuse=tf_compat.AUTO_REUSE):
        sec_settings = [
            ResNetSection(num_blocks=2, out_channels=16, downsample=False),
            ResNetSection(num_blocks=2, out_channels=32, downsample=True),
            ResNetSection(num_blocks=2, out_channels=64, downsample=True),
        ]
        net = resnet_const(
            inputs,
            training,
            sec_settings,
            num_classes,
            class_type=class_type,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            simplified_arch=True,
        )

    return net


@ModelRegistry.register(
    key=["resnet34", "resnet_34", "resnet-34", "resnetv1_34", "resnetv1-34"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="34",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def resnet34(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:
    """
    Standard ResNet34 implementation;
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
        ResNetSection(num_blocks=3, out_channels=64, downsample=False),
        ResNetSection(num_blocks=4, out_channels=128, downsample=True),
        ResNetSection(num_blocks=6, out_channels=256, downsample=True),
        ResNetSection(num_blocks=3, out_channels=512, downsample=True),
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
    key=["resnet50", "resnet_50", "resnet-50", "resnetv1_50", "resnetv1-50"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="resnet_v1",
    sub_architecture="50",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def resnet50(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:
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
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def resnet101(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:
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
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def resnet152(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = None,
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:
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
