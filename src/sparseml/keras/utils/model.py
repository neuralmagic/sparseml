"""
Utils for Keras model
"""

import tensorflow as tf
from tensorflow import keras


__all__ = ["sparsity"]


def sparsity(model: keras.Model):
    """
    Retrieve sparsity of a Keras model

    :param model: a Keras model
    :return: (1) model sparsity, (2) dictionary of layer sparsity
    """
    zero = tf.constant(0, dtype=tf.float32)
    model_weight_size = 0
    model_zeros = 0
    sparsity_dict = {}

    for layer in model.layers:
        layer_sparsity_dict = {}

        for i, weight in enumerate(layer.trainable_weights):
            weight_name = (
                weight.name if hasattr(weight, "name") else "weight_{}".format(i)
            )
            mask = tf.cast(tf.equal(weight, zero), tf.uint8)

            weight_size = tf.size(weight)
            zeros = tf.cast(tf.math.count_nonzero(mask), tf.int32)
            layer_sparsity_dict[weight.name] = zeros / weight_size

            model_weight_size += weight_size
            model_zeros += zeros

            sparsity_dict[layer.name] = layer_sparsity_dict

    model_sparsity = model_zeros / model_weight_size

    return model_sparsity, sparsity_dict
