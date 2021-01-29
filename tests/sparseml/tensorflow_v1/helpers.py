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

from sparseml.tensorflow_v1.utils import tf_compat


__all__ = ["mlp_net", "conv_net"]


def _conv(name, x_tens, in_chan, out_chan, kernel, stride, padding, add_relu=True):
    """
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/conv2d
    """

    with tf_compat.name_scope(name):
        weight = tf_compat.Variable(
            tf_compat.random_normal([kernel, kernel, in_chan, out_chan]), name="weight"
        )
        bias = tf_compat.Variable(tf_compat.random_normal([out_chan]), name="bias")

        x_tens = tf_compat.nn.conv2d(
            x_tens, weight, strides=[1, stride, stride, 1], padding=padding, name="conv"
        )
        x_tens = tf_compat.nn.bias_add(x_tens, bias, name="add")

        if add_relu:
            x_tens = tf_compat.nn.relu(x_tens, name="relu")

    return x_tens


def _fc(name, x_tens, in_chan, out_chan, add_relu=True):
    with tf_compat.name_scope(name):
        weight = tf_compat.Variable(
            tf_compat.random_normal([in_chan, out_chan]), name="weight"
        )
        bias = tf_compat.Variable(tf_compat.random_normal([out_chan]), "bias")

        x_tens = tf_compat.matmul(x_tens, weight, name="matmul")
        x_tens = tf_compat.add(x_tens, bias, name="add")

        if add_relu:
            x_tens = tf_compat.nn.relu(x_tens, name="relu")

    return x_tens


def mlp_net():
    inp = tf_compat.placeholder(tf_compat.float32, [None, 16], name="inp")

    with tf_compat.name_scope("mlp_net"):
        fc1 = _fc("fc1", inp, 16, 32)
        fc2 = _fc("fc2", fc1, 32, 64)
        fc3 = _fc("fc3", fc2, 64, 64, add_relu=False)

    out = tf_compat.sigmoid(fc3, name="out")

    return out, inp


def conv_net():
    inp = tf_compat.placeholder(tf_compat.float32, [None, 28, 28, 1], name="inp")

    with tf_compat.name_scope("conv_net"):
        conv1 = _conv("conv1", inp, 1, 32, 3, 2, "SAME")
        conv2 = _conv("conv2", conv1, 32, 32, 3, 2, "SAME")
        avg_pool = tf_compat.reduce_mean(conv2, axis=[1, 2])
        reshape = tf_compat.reshape(avg_pool, [-1, 32])
        mlp = _fc("mlp", reshape, 32, 10, add_relu=False)

    out = tf_compat.sigmoid(mlp, name="out")

    return out, inp
