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
PyTorch VGG implementations.
Further info can be found in the paper `here <https://arxiv.org/abs/1409.1556>`__.
"""

from typing import List, Union

from sparseml.tensorflow_v1.models.estimator import ClassificationEstimatorModelFn
from sparseml.tensorflow_v1.models.registry import ModelRegistry
from sparseml.tensorflow_v1.nn import conv2d_block, dense_block, pool2d
from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "VGGSection",
    "vgg_const",
    "vgg11",
    "vgg11bn",
    "vgg13",
    "vgg13bn",
    "vgg16",
    "vgg16bn",
    "vgg19",
    "vgg19bn",
]


BASE_NAME_SCOPE = "vgg"


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
        if num_classes:
            if class_type:
                if class_type == "single":
                    final_act = "softmax"
                elif class_type == "multi":
                    final_act = "sigmoid"
                else:
                    raise ValueError(
                        "unknown class_type given of {}".format(class_type)
                    )
            else:
                final_act = None

            out = tf_compat.transpose(x_tens, [0, 3, 1, 2])
            out = tf_compat.reshape(out, [-1, 7 * 7 * 512])
            out = dense_block(
                "mlp_0",
                out,
                training,
                channels=4096,
                include_bn=False,
                include_bias=True,
                dropout_rate=0.5,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
            )
            out = dense_block(
                "mlp_1",
                out,
                training,
                channels=4096,
                include_bn=False,
                include_bias=True,
                dropout_rate=0.5,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
            )
            logits = dense_block(
                "mlp_2",
                out,
                training,
                channels=num_classes,
                include_bn=False,
                include_bias=True,
                act=final_act,
                dropout_rate=0.5,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
            )
        else:
            logits = x_tens

    return logits


class VGGSection(object):
    """
    Settings to describe how to put together a VGG architecture
    using user supplied configurations.

    :param num_blocks: the number of blocks to put in the section (conv [bn] relu)
    :param out_channels: the number of output channels from the section
    :param use_batchnorm: True to put batchnorm after each conv, False otherwise
    """

    def __init__(self, num_blocks: int, out_channels: int, use_batchnorm: bool):
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm

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
            for block in range(self.num_blocks):
                out = conv2d_block(
                    name="block_{}".format(block),
                    x_tens=out,
                    training=training,
                    channels=self.out_channels,
                    kernel_size=3,
                    include_bn=self.use_batchnorm,
                    include_bias=True,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    beta_initializer=beta_initializer,
                    gamma_initializer=gamma_initializer,
                )

            out = pool2d(
                name="pool",
                x_tens=out,
                type_="max",
                pool_size=2,
                strides=2,
                padding="valid",
            )

        return out


def vgg_const(
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    sec_settings: List[VGGSection],
    num_classes: int,
    class_type: str,
    kernel_initializer,
    bias_initializer,
    beta_initializer,
    gamma_initializer,
) -> tf_compat.Tensor:
    """
    Graph constructor for VGG implementation.

    :param x_tens: The input tensor to the VGG architecture
    :param training: bool or Tensor to specify if the model should be run
        in training or inference mode
    :param sec_settings: The settings for each section in the VGG modoel
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
    out = x_tens

    with tf_compat.variable_scope(BASE_NAME_SCOPE, reuse=tf_compat.AUTO_REUSE):
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
    key=["vgg11", "vgg_11", "vgg-11"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="11",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg11(
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
    Standard VGG 11 implementation;
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
        VGGSection(num_blocks=1, out_channels=64, use_batchnorm=False),
        VGGSection(num_blocks=1, out_channels=128, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=256, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=False),
    ]

    return vgg_const(
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
    key=["vgg11bn", "vgg_11bn", "vgg-11bn"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="11-bn",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg11bn(
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
    Standard VGG 11 batch normalized implementation;
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
        VGGSection(num_blocks=1, out_channels=64, use_batchnorm=True),
        VGGSection(num_blocks=1, out_channels=128, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=256, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=True),
    ]

    return vgg_const(
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
    key=["vgg13", "vgg_13", "vgg-13"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="13",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg13(
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
    Standard VGG 13 implementation;
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
        VGGSection(num_blocks=2, out_channels=64, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=128, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=256, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=False),
    ]

    return vgg_const(
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
    key=["vgg13bn", "vgg_13bn", "vgg-13bn"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="13-bn",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg13bn(
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
    Standard VGG 13 batch normalized implementation;
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
        VGGSection(num_blocks=2, out_channels=64, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=128, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=256, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=512, use_batchnorm=True),
    ]

    return vgg_const(
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
    key=["vgg16", "vgg_16", "vgg-16"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="16",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg16(
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
    Standard VGG 16 implementation;
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
        VGGSection(num_blocks=2, out_channels=64, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=128, use_batchnorm=False),
        VGGSection(num_blocks=3, out_channels=256, use_batchnorm=False),
        VGGSection(num_blocks=3, out_channels=512, use_batchnorm=False),
        VGGSection(num_blocks=3, out_channels=512, use_batchnorm=False),
    ]

    return vgg_const(
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
    key=["vgg16bn", "vgg_16bn", "vgg-16bn"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="16-bn",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg16bn(
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
    Standard VGG 16 batch normalized implementation;
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
        VGGSection(num_blocks=2, out_channels=64, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=128, use_batchnorm=True),
        VGGSection(num_blocks=3, out_channels=256, use_batchnorm=True),
        VGGSection(num_blocks=3, out_channels=512, use_batchnorm=True),
        VGGSection(num_blocks=3, out_channels=512, use_batchnorm=True),
    ]

    return vgg_const(
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
    key=["vgg19", "vgg_19", "vgg-19"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="19",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg19(
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
    Standard VGG 19 implementation;
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
        VGGSection(num_blocks=2, out_channels=64, use_batchnorm=False),
        VGGSection(num_blocks=2, out_channels=128, use_batchnorm=False),
        VGGSection(num_blocks=4, out_channels=256, use_batchnorm=False),
        VGGSection(num_blocks=4, out_channels=512, use_batchnorm=False),
        VGGSection(num_blocks=4, out_channels=512, use_batchnorm=False),
    ]

    return vgg_const(
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
    key=["vgg19bn", "vgg_19bn", "vgg-19bn"],
    input_shape=(224, 224, 3),
    domain="cv",
    sub_domain="classification",
    architecture="vgg",
    sub_architecture="19-bn",
    default_dataset="imagenet",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[".+/classifier/mlp_2/fc/.+"],
)
def vgg19bn(
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
    Standard VGG 19 batch normalized implementation;
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
        VGGSection(num_blocks=2, out_channels=64, use_batchnorm=True),
        VGGSection(num_blocks=2, out_channels=128, use_batchnorm=True),
        VGGSection(num_blocks=4, out_channels=256, use_batchnorm=True),
        VGGSection(num_blocks=4, out_channels=512, use_batchnorm=True),
        VGGSection(num_blocks=4, out_channels=512, use_batchnorm=True),
    ]

    return vgg_const(
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
