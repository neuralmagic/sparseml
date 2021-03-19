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
Utils for Keras model
"""

import tensorflow

from sparseml.keras.utils import keras


__all__ = ["sparsity"]


def sparsity(model: keras.Model):
    """
    Retrieve sparsity of a Keras model

    :param model: a Keras model
    :return: (1) model sparsity, (2) dictionary of layer sparsity
    """
    zero = tensorflow.constant(0, dtype=tensorflow.float32)
    model_weight_size = 0
    model_zeros = 0
    sparsity_dict = {}

    for layer in model.layers:
        layer_sparsity_dict = {}

        for i, weight in enumerate(layer.trainable_weights):
            mask = tensorflow.cast(tensorflow.equal(weight, zero), tensorflow.uint8)

            weight_size = tensorflow.size(weight)
            zeros = tensorflow.cast(
                tensorflow.math.count_nonzero(mask), tensorflow.int32
            )
            layer_sparsity_dict[weight.name] = zeros / weight_size

            model_weight_size += weight_size
            model_zeros += zeros

            sparsity_dict[layer.name] = layer_sparsity_dict

    model_sparsity = model_zeros / model_weight_size

    return model_sparsity, sparsity_dict
