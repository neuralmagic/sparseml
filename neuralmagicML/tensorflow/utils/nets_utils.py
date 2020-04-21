"""
Utility functions for working with tensorflow slim's nets_factory

=================================================================

from neuralmagicML.tensorflow.utils import nets_utils
"""

import functools
from typing import Callable, Dict

from neuralmagicML.tensorflow.utils import tf_compat as tf

from nets import nets_factory
from tensorflow.contrib import slim
from tensorflow.contrib import layers as contrib_layers


def get_network_fn(
    name: str,
    num_classes: int,
    weight_decay: float = 0.0,
    is_training: bool = False,
    arg_scope_vars: Dict = {},
):
    """
    Modified from slim/nets/nets_factory
    Returns a network_fn such as `logits, end_points = network_fn(images)`.

    :param name: The name of the network.
    :param num_classes: The number of classes to use for classification. If 0 or None,
     the logits layer is omitted and its input features are returned instead.
    :param weight_decay: The l2 coefficient for the model weights.
    :param is_training: `True` if the model is being used for training otherwise `False`
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
    :raise: ValueError If network `name` is not recognized.
  """
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


def get_model_scope(model_name: str, arg_scope_vars: Dict = {}):
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
    normalizer_fn: Callable = slim.batch_norm,
):
    """Adapted from slim to allow for Xavier initializer
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
