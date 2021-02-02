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
Utility functions for working with tensorflow_v1 slim's nets_factory
"""

import functools
import logging
from typing import Callable, Dict

from sparseml.tensorflow_v1.utils import tf_compat as tf


try:
    from nets import cyclegan, dcgan, nets_factory
except Exception:
    nets_factory = None
    dcgan = None
    cyclegan = None
    logging.warning("TensorFlow slim nets not found in system")

try:
    from tensorflow.contrib import layers as contrib_layers
    from tensorflow.contrib import slim
except Exception:
    slim = None
    contrib_layers = None
    logging.warning("TensorFlow slim not found in system")


__all__ = [
    "get_network_fn",
    "get_gan_network_fn",
    "get_model_scope",
    "mobilenet_v1_arg_scope",
]


def _gans_constructors() -> Dict[str, Callable]:
    return {
        "cyclegan": cyclegan.cyclegan_generator_resnet,
        "dcgan_generator": dcgan.generator,
        "dcgan_discriminator": dcgan.discriminator,
    }


def _check_slim_availability():
    if nets_factory is None or slim is None:
        raise ValueError(
            "TensorFlow slim not setup in environment, please install first"
        )


def get_network_fn(
    name: str,
    num_classes: int,
    weight_decay: float = 0.0,
    is_training: bool = False,
    arg_scope_vars: Dict = None,
):
    """
    Modified from slim/nets/nets_factory
    Returns a network_fn such as `logits, end_points = network_fn(images)`.

    :param name: The name of the network.
    :param num_classes: The number of classes to use for classification. If 0 or None,
        the logits layer is omitted and its input features are returned instead.
    :param weight_decay: The l2 coefficient for the model weights.
    :param is_training: `True` if the model is being used for training otherwise `False`
    :param arg_scope_vars: arg_scope_vars to be passed to the slim arg_scope
    :return network_fn: A function that applies the model to a batch of images. It has
        the following signature: net, end_points = network_fn(images)
        The `images` input is a tensor of shape [batch_size, height, width, 3 or
        1] with height = width = network_fn.default_image_size. (The
        permissibility and treatment of other sizes depends on the network_fn.)
        The returned `end_points` are a dictionary of intermediate activations.
        The returned `net` is the topmost layer, depending on `num_classes`:
        If `num_classes` was a non-zero integer, `net` is a logits tensor
        of shape [batch_size, num_classes].
        If `num_classes` was 0 or `None`, `net` is a tensor with the input
        to the logits layer of shape [batch_size, 1, 1, num_features] or
        [batch_size, num_features]. Dropout has not been applied to this
        (even if the network's original classification does); it remains for
        the caller to do this or not.
    :raises ValueError: If network `name` is not recognized.
    """
    _check_slim_availability()

    if not arg_scope_vars:
        arg_scope_vars = {}

    if "gan" in name.lower():
        return get_gan_network_fn(name, is_training)
    if name not in nets_factory.networks_map:
        raise ValueError("Name of network unknown %s" % name)
    func = nets_factory.networks_map[name]
    arg_scope_vars["weight_decay"] = weight_decay

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(get_model_scope(name, arg_scope_vars=arg_scope_vars)):
            return func(
                images, num_classes=num_classes, is_training=is_training, **kwargs
            )

    if hasattr(func, "default_image_size"):
        network_fn.default_image_size = func.default_image_size

    return network_fn


def get_gan_network_fn(
    name: str,
    is_training: bool = False,
):
    """
    Returns network_fn for a GAN sub-model

    :param name: The name of the network.
    :param is_training: `True` if the model is being used for training otherwise `False`
    :return network_fn: Function that will run a gan sub-model
    :raises ValueError: If network `name` is not recognized.
    """
    _check_slim_availability()

    if name not in _gans_constructors():
        raise ValueError("Name of GAN network unknown %s" % name)

    func = _gans_constructors()[name]

    def network_fn(inputs, **kwargs):
        if name == "dcgan_generator":
            kwargs["final_size"] = 16
        return func(inputs, is_training=is_training, **kwargs)

    return network_fn


def get_model_scope(model_name: str, arg_scope_vars: Dict = None):
    """
    :param model_name: name of the model to create an arg scope for
    :param arg_scope_vars:
    :return: arg_scope_vars to be passed to the slim arg_scope
    """
    _check_slim_availability()

    if arg_scope_vars is None:
        arg_scope_vars = {}

    arg_scope = nets_factory.arg_scopes_map[model_name](**arg_scope_vars)
    if model_name == "mobilenet_v1":
        arg_scope = mobilenet_v1_arg_scope(**arg_scope_vars)
    return arg_scope


def mobilenet_v1_arg_scope(
    is_training: bool = True,
    weight_decay: float = 0.00004,
    stddev: float = 0.09,
    regularize_depthwise: bool = False,
    batch_norm_decay: float = 0.9997,
    batch_norm_epsilon: float = 0.001,
    batch_norm_updates_collections: tf.GraphKeys = tf.GraphKeys.UPDATE_OPS,
    normalizer_fn: Callable = slim.batch_norm if slim else None,
):
    """
    Adapted from slim to allow for Xavier initializer
    Defines the default MobilenetV1 arg scope.

    :param is_training: Whether or not we're training the model. If this is set to
        None, the parameter is not added to the batch_norm arg_scope.
    :param weight_decay: The weight decay to use for regularizing the model.
    :param stddev: The standard deviation of the trunctated normal weight initializer.
    :param regularize_depthwise: Whether or not apply regularization on depthwise.
    :param batch_norm_decay: Decay for batch norm moving average.
    :param batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
    :param batch_norm_updates_collections: Collection for the update ops for
        batch norm.
    :param normalizer_fn: Normalization function to apply after convolution.
    :return: An `arg_scope` to use for the mobilenet v1 model.
    """
    _check_slim_availability()

    batch_norm_params = {
        "center": True,
        "scale": True,
        "decay": batch_norm_decay,
        "epsilon": batch_norm_epsilon,
        "updates_collections": batch_norm_updates_collections,
    }
    if is_training is not None:
        batch_norm_params["is_training"] = is_training

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.keras.initializers.glorot_normal()
    regularizer = contrib_layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_initializer=weights_init,
        activation_fn=tf.nn.relu6,
        normalizer_fn=normalizer_fn,
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope(
                    [slim.separable_conv2d], weights_regularizer=depthwise_regularizer
                ) as sc:
                    return sc
