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
TensorFlow MobileNet V2 implementations.
Further info can be found in the paper `here <https://arxiv.org/abs/1801.04381>`__.
"""

from typing import List, Union

from sparseml.tensorflow_v1.models.estimator import ClassificationEstimatorModelFn
from sparseml.tensorflow_v1.models.registry import ModelRegistry
from sparseml.tensorflow_v1.nn import (
    conv2d_block,
    dense_block,
    depthwise_conv2d_block,
    pool2d,
)
from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "MobileNetV2Section",
    "mobilenet_v2_const",
    "mobilenet_v2_width",
    "mobilenet_v2",
]


BASE_NAME_SCOPE = "mobilenet_v2"


def _make_divisible(
    value: float, divisor: int, min_value: Union[int, None] = None
) -> int:
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    if new_value < 0.9 * value:
        new_value += divisor

    return new_value


def _input_inverted_bottleneck_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    out_channels: int,
    exp_channels: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = conv2d_block(
            "expand",
            x_tens,
            training,
            exp_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            act="relu6",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = depthwise_conv2d_block(
            "spatial",
            out,
            training,
            exp_channels,
            kernel_size=3,
            padding="same",
            stride=1,
            act="relu6",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = conv2d_block(
            "compress",
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

    return out


def _inverted_bottleneck_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    out_channels: int,
    exp_channels: int,
    stride: int,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = conv2d_block(
            "expand",
            x_tens,
            training,
            exp_channels,
            kernel_size=1,
            act="relu6",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = depthwise_conv2d_block(
            "spatial",
            out,
            training,
            exp_channels,
            kernel_size=3,
            stride=stride,
            act="relu6",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = conv2d_block(
            "compress",
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

        if stride == 1 and int(x_tens.shape[3]) == out_channels:
            out = tf_compat.add(out, x_tens)

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
            logits = tf_compat.layers.dropout(
                logits, 0.2, training=training, name="dropout"
            )
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


class MobileNetV2Section(object):
    """
    Settings to describe how to put together MobileNet V2 architecture
    using user supplied configurations.

    :param num_blocks: the number of inverted bottleneck blocks to put in the section
    :param out_channels: the number of output channels from the section
    :param downsample: True to apply stride 2 for down sampling of the input,
        False otherwise
    :param exp_channels: number of channels to expand out to,
        if not supplied uses exp_ratio
    :param exp_ratio: the expansion ratio to use for the depthwise convolution
    :param init_section: True if it is the initial section, False otherwise
    :param width_mult: The width multiplier to apply to the channel sizes
    """

    def __init__(
        self,
        num_blocks: int,
        out_channels: int,
        downsample: bool,
        exp_channels: Union[None, int] = None,
        exp_ratio: float = 1.0,
        init_section: bool = False,
        width_mult: float = 1.0,
    ):
        self.num_blocks = num_blocks
        self.out_channels = _make_divisible(out_channels * width_mult, 8)
        self.exp_channels = exp_channels
        self.exp_ratio = exp_ratio
        self.downsample = downsample
        self.init_section = init_section

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
        :param x_tens: The input tensor to the MobileNet architecture
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
            exp_channels = (
                self.exp_channels
                if self.exp_channels is not None
                else _make_divisible(int(out.shape[3]) * self.exp_ratio, 8)
            )

            for block in range(self.num_blocks):
                if self.init_section and block == 0:
                    out = _input_inverted_bottleneck_block(
                        name="block_{}".format(block),
                        x_tens=out,
                        training=training,
                        out_channels=self.out_channels,
                        exp_channels=exp_channels,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        beta_initializer=beta_initializer,
                        gamma_initializer=gamma_initializer,
                    )
                else:
                    out = _inverted_bottleneck_block(
                        name="block_{}".format(block),
                        x_tens=out,
                        training=training,
                        out_channels=self.out_channels,
                        exp_channels=exp_channels,
                        stride=stride,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        beta_initializer=beta_initializer,
                        gamma_initializer=gamma_initializer,
                    )

                stride = 1
                exp_channels = (
                    self.exp_channels
                    if self.exp_channels is not None
                    else _make_divisible(self.out_channels * self.exp_ratio, 8)
                )

        return out


def mobilenet_v2_const(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    sec_settings: List[MobileNetV2Section],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    """
    Graph constructor for MobileNet V2 implementation.

    :param x_tens: The input tensor to the MobileNet architecture
    :param training: bool or Tensor to specify if the model should be run
        in training or inference mode
    :param sec_settings: The settings for each section in the MobileNet modoel
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

    with tf_compat.variable_scope(BASE_NAME_SCOPE, reuse=tf_compat.AUTO_REUSE):
        out = x_tens

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

        out = conv2d_block(
            name="feat_extraction",
            x_tens=out,
            training=training,
            channels=1280,
            kernel_size=1,
            act=None,
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


def mobilenet_v2_width(
    width_mult: float,
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
    Standard MobileNetV2 implementation for a given width;
    expected input shape is (B, 224, 224, 3)

    :param width_mult: The width multiplier for the architecture to create.
        1.0 is standard, 0.5 is half the size, 2.0 is twice the size.
    :param inputs: The input tensor to the MobileNet architecture
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
        MobileNetV2Section(
            num_blocks=1,
            out_channels=16,
            exp_channels=32,
            downsample=False,
            init_section=True,
            width_mult=width_mult,
        ),
        MobileNetV2Section(
            num_blocks=2,
            out_channels=24,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobileNetV2Section(
            num_blocks=3,
            out_channels=32,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobileNetV2Section(
            num_blocks=4,
            out_channels=64,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobileNetV2Section(
            num_blocks=3,
            out_channels=96,
            exp_ratio=6,
            downsample=False,
            init_section=False,
            width_mult=width_mult,
        ),
        MobileNetV2Section(
            num_blocks=3,
            out_channels=160,
            exp_ratio=6,
            downsample=True,
            init_section=False,
            width_mult=width_mult,
        ),
        MobileNetV2Section(
            num_blocks=1,
            out_channels=320,
            exp_ratio=6,
            downsample=False,
            init_section=False,
            width_mult=width_mult,
        ),
    ]

    return mobilenet_v2_const(
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
    key=[
        "mobilenetv2",
        "mobilenet_v2",
        "mobilenet_v2_100",
        "mobilenet-v2",
        "mobilenet-v2-100",
        "mobilenetv2_1.0",
    ],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="mobilenet_v2",
    sub_architecture="1.0",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def mobilenet_v2(
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
    Standard MobileNet V2 implementation with width=1.0;
    expected input shape is (B, 224, 224, 3)

    :param inputs: The input tensor to the MobileNet architecture
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
    return mobilenet_v2_width(
        1.0,
        inputs,
        training,
        num_classes,
        class_type,
        kernel_initializer,
        bias_initializer,
        beta_initializer,
        gamma_initializer,
    )
