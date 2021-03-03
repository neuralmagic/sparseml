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
TensorFlow MobileNet implementations.
Further info can be found in the paper `here <https://arxiv.org/abs/1704.04861>`__.
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
    "MobileNetSection",
    "mobilenet_const",
    "mobilenet",
]


BASE_NAME_SCOPE = "mobilenet"


def _input(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    kernel_initializer,
    bias_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    out = conv2d_block(
        "input",
        x_tens,
        training,
        channels=32,
        kernel_size=3,
        stride=2,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        gamma_initializer=gamma_initializer,
    )

    return out


def _dw_sep_block(
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
        out = depthwise_conv2d_block(
            "depth",
            x_tens,
            training,
            int(x_tens.shape[3]),
            kernel_size=3,
            padding=1,
            stride=stride,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            gamma_initializer=gamma_initializer,
        )
        out = conv2d_block(
            "point",
            out,
            training,
            out_channels,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
        )

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


class MobileNetSection(object):
    """
    Settings to describe how to put together a MobileNet architecture
    using user supplied configurations.

    :param num_blocks: the number of depthwise separable blocks to put in the section
    :param out_channels: the number of output channels from the section
    :param downsample: True to apply stride 2 for down sampling of the input,
        False otherwise
    """

    def __init__(self, num_blocks: int, out_channels: int, downsample: bool):
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.downsample = downsample

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

            for block in range(self.num_blocks):
                out = _dw_sep_block(
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


def mobilenet_const(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    sec_settings: List[MobileNetSection],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    """
    Graph constructor for MobileNet implementation.

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
        out = _input(
            x_tens, training, kernel_initializer, bias_initializer, gamma_initializer
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
    key=[
        "mobilenet",
        "mobilenet_100",
        "mobilenet-v1",
        "mobilenet-v1-100",
        "mobilenet_v1",
        "mobilenet_v1_100",
        "mobilenetv1_1.0",
    ],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="mobilenet_v1",
    sub_architecture="1.0",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/dense/fc/.+"],
)
def mobilenet(
    inputs: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor] = True,
    num_classes: int = 1000,
    class_type: str = "single",
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
) -> tf_compat.Tensor:
    """
    Standard MobileNet implementation with width=1.0;
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
    sec_settings = [
        MobileNetSection(num_blocks=1, out_channels=64, downsample=False),
        MobileNetSection(num_blocks=2, out_channels=128, downsample=True),
        MobileNetSection(num_blocks=2, out_channels=256, downsample=True),
        MobileNetSection(num_blocks=6, out_channels=512, downsample=True),
        MobileNetSection(num_blocks=2, out_channels=1024, downsample=True),
    ]

    return mobilenet_const(
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
