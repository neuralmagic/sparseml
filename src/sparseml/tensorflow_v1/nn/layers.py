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

from typing import Tuple, Union

from sparseml.tensorflow_v1.utils import tf_compat


__all__ = [
    "activation",
    "pool2d",
    "conv2d_block",
    "depthwise_conv2d_block",
    "dense_block",
    "fc",
    "conv2d",
]


BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5


def activation(x_tens: tf_compat.Tensor, act: Union[None, str], name: str = "act"):
    """
    Create an activation operation in the current graph and scope.

    :param x_tens: the tensor to apply the op to
    :param act: the activation type to apply, supported:
        [None, relu, relu6, sigmoid, softmax]
    :param name: the name to give to the activation op in the graph
    :return: the created operation
    """
    if not act:
        return x_tens

    if act == "relu":
        return tf_compat.nn.relu(x_tens, name=name)

    if act == "relu6":
        return tf_compat.nn.relu6(x_tens, name=name)

    if act == "sigmoid":
        return tf_compat.nn.sigmoid(x_tens, name=name)

    if act == "softmax":
        return tf_compat.nn.softmax(x_tens, name=name)

    raise ValueError("unknown act given of {}".format(act))


def symmetric_pad2d(
    x_tens: tf_compat.Tensor, pad: Union[str, int, Tuple[int, int]], data_format: str
):
    """
    Create a symmetric pad op in the current graph and scope.
    To do this, pad must be an integer or tuple of integers.
    If pad is a string, will not do anything and pad should be passed into
    the pool or conv op.

    :param x_tens: the tensor to apply padding to
    :param pad: the padding to apply symmetrically. If it is a single integer,
        will apply to both sides of height and width dimensions.
        If it is a tuple, will take the first element as the padding for
        both sides of height dimensions and second for booth sides of width ddimension.
    :param data_format: either channels_last or channels_first
    :return: the padded tensor
    """
    if isinstance(pad, str):
        # default tensorflow_v1 padding
        return x_tens

    y_pad = [pad, pad] if isinstance(pad, int) else [pad[0], pad[0]]
    x_pad = [pad, pad] if isinstance(pad, int) else [pad[1], pad[1]]
    pad_tensor = (
        [[0, 0], y_pad, x_pad, [0, 0]]
        if data_format == "channels_last"
        else [[0, 0], [0, 0], y_pad, x_pad]
    )
    pad_tensor = tf_compat.constant(pad_tensor)

    return tf_compat.pad(x_tens, pad_tensor)


def pool2d(
    name: str,
    x_tens: tf_compat.Tensor,
    type_: str,
    pool_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, ...]] = "same",
    data_format: str = "channels_last",
):
    """
    Create a pool op with the given name in the current graph and scope.
    Supported are [max, avg, global_avg]

    :param name: the name to given to the pooling op in the graph
    :param x_tens: the input tensor to apply pooling to
    :param type_: the type of pooling to apply, one of [max, avg, global_avg]
    :param pool_size: the size of the pooling window to apply,
        if global_avg then is the desired output size
    :param strides: the stride to apply for the pooling op,
        if global_avg then is unused
    :param padding: any padding to apply to the tensor before pooling;
        if string then uses tensorflows built in padding, else uses symmetric_pad2d
    :param data_format: either channels_last or channels_first
    :return: the tensor after pooling
    """
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = symmetric_pad2d(x_tens, padding, data_format)

        if type_ == "max":
            return tf_compat.layers.max_pooling2d(
                out,
                pool_size,
                strides,
                padding if isinstance(padding, str) else "valid",
                data_format,
            )
        elif type_ == "avg":
            return tf_compat.layers.average_pooling2d(
                out,
                pool_size,
                strides,
                padding if isinstance(padding, str) else "valid",
                data_format,
            )
        elif type_ == "global_avg":
            if pool_size != 1 and pool_size != (1, 1):
                raise ValueError(
                    "only output pool_size of 1 is supported for global average pooling"
                )

            return tf_compat.reduce_mean(
                out,
                [1, 2] if data_format == "channels_last" else [2, 3],
                keepdims=True,
            )
        else:
            raise ValueError("unrecognized type_ given of {}".format(type_))


def conv2d_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    channels: int,
    kernel_size: int,
    padding: Union[str, int, Tuple[int, ...]] = "same",
    stride: int = 1,
    data_format: str = "channels_last",
    include_bn: bool = True,
    include_bias: bool = None,
    act: Union[None, str] = "relu",
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
):
    """
    Create a convolution op and supporting ops (batch norm, activation, etc)
    in the current graph and scope.

    :param name: The name to group all ops under in the graph
    :param x_tens: The input tensor to apply a convolution and supporting ops to
    :param training: A bool or tensor to indicate if the net is being run
        in training mode or not. Used for batch norm
    :param channels: The number of output channels from the conv op
    :param kernel_size: The size of the kernel to use for the conv op
    :param padding: Any padding to apply to the tensor before the convolution;
        if string then uses tensorflows built in padding, else uses symmetric_pad2d
    :param stride: The stride to apply for the convolution
    :param data_format: Either channels_last or channels_first
    :param include_bn: True to include a batch norm operation after the conv,
        False otherwise
    :param include_bias: If left unset, will add a bias if not include_bn.
        Otherwise can be set to True to include a bias after the convolution,
        False otherwise.
    :param act: The activation to apply after the conv op and batch norm (if included).
        Default is "relu", set to None for no activation.
    :param kernel_initializer: The initializer to use for the convolution kernels
    :param bias_initializer: The initializer to use for the bias variable,
        if a bias is included
    :param beta_initializer: The initializer to use for the beta variable,
        if batch norm is included
    :param gamma_initializer: The initializer to use for the gamma variable,
        if gamma is included
    :return: the tensor after all ops have been applied
    """
    if include_bias is None:
        include_bias = not include_bn

    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = symmetric_pad2d(x_tens, padding, data_format)
        out = tf_compat.layers.conv2d(
            out,
            filters=channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding if isinstance(padding, str) else "valid",
            data_format=data_format,
            use_bias=include_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer if include_bias else None,
            name="conv",
        )

        if include_bn:
            out = tf_compat.layers.batch_normalization(
                out,
                axis=1 if data_format == "channels_first" else 3,
                momentum=BN_MOMENTUM,
                epsilon=BN_EPSILON,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                training=training,
                name="bn",
            )

        out = activation(out, act)

    return out


def depthwise_conv2d_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    channels: int,
    kernel_size: int,
    padding: Union[str, int, Tuple[int, ...]] = "same",
    stride: int = 1,
    data_format: str = "channels_last",
    include_bn: bool = True,
    include_bias: bool = None,
    act: Union[None, str] = "relu",
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
):
    """
    Create a depthwise convolution op and supporting ops (batch norm, activation, etc)
    in the current graph and scope.

    :param name: The name to group all ops under in the graph
    :param x_tens: The input tensor to apply a convolution and supporting ops to
    :param training: A bool or tensor to indicate if the net is being run
        in training mode or not. Used for batch norm
    :param channels: The number of output channels from the conv op
    :param kernel_size: The size of the kernel to use for the conv op
    :param padding: Any padding to apply to the tensor before the convolution;
        if string then uses tensorflows built in padding, else uses symmetric_pad2d
    :param stride: The stride to apply for the convolution
    :param data_format: Either channels_last or channels_first
    :param include_bn: True to include a batch norm operation after the conv,
        False otherwise
    :param include_bias: If left unset, will add a bias if not include_bn.
        Otherwise can be set to True to include a bias after the convolution,
        False otherwise.
    :param act: The activation to apply after the conv op and batch norm (if included).
        Default is "relu", set to None for no activation.
    :param kernel_initializer: The initializer to use for the convolution kernels
    :param bias_initializer: The initializer to use for the bias variable,
        if a bias is included
    :param beta_initializer: The initializer to use for the beta variable,
        if batch norm is included
    :param gamma_initializer: The initializer to use for the gamma variable,
        if gamma is included
    :return: the tensor after all ops have been applied
    """
    if include_bias is None:
        include_bias = not include_bn

    channel_axis = 3 if data_format == "channels_last" else 1
    stride = (
        [1, stride, stride, 1]
        if data_format == "channels_last"
        else [1, stride, stride, 1]
    )
    kernel_shape = (kernel_size, kernel_size, int(x_tens.shape[channel_axis]), 1)

    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        with tf_compat.variable_scope("conv"):
            kernel = tf_compat.get_variable(
                "kernel",
                shape=kernel_shape,
                initializer=kernel_initializer,
                trainable=True,
            )
            bias = (
                tf_compat.get_variable(
                    "bias",
                    shape=(channels,),
                    initializer=bias_initializer,
                    trainable=True,
                )
                if include_bias
                else None
            )

            out = symmetric_pad2d(x_tens, padding, data_format)
            out = tf_compat.nn.depthwise_conv2d(
                out,
                kernel,
                stride,
                padding=padding.upper() if isinstance(padding, str) else "VALID",
                data_format="NHWC" if data_format == "channels_last" else "NCHW",
            )

            if bias is not None:
                out = tf_compat.nn.bias_add(out, bias, data_format)

        if include_bn:
            out = tf_compat.layers.batch_normalization(
                out,
                axis=3 if data_format == "channels_last" else 1,
                momentum=BN_MOMENTUM,
                epsilon=BN_EPSILON,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                training=training,
                name="bn",
            )

        out = activation(out, act)

    return out


def dense_block(
    name: str,
    x_tens: tf_compat.Tensor,
    training: Union[bool, tf_compat.Tensor],
    channels: int,
    include_bn: bool = False,
    include_bias: bool = None,
    dropout_rate: float = None,
    act: Union[None, str] = "relu",
    kernel_initializer=tf_compat.glorot_uniform_initializer(),
    bias_initializer=tf_compat.zeros_initializer(),
    beta_initializer=tf_compat.zeros_initializer(),
    gamma_initializer=tf_compat.ones_initializer(),
):
    """
    Create a dense or fully connected op and supporting ops
    (batch norm, activation, etc) in the current graph and scope.

    :param name: The name to group all ops under in the graph
    :param x_tens: The input tensor to apply a fully connected and supporting ops to
    :param training: A bool or tensor to indicate if the net is being run
        in training mode or not. Used for batch norm and dropout
    :param channels: The number of output channels from the dense op
    :param include_bn: True to include a batch norm operation after the conv,
        False otherwise
    :param include_bias: If left unset, will add a bias if not include_bn.
        Otherwise can be set to True to include a bias after the convolution,
        False otherwise.
    :param dropout_rate: The dropout rate to apply after the fully connected
        and batch norm if included. If none, will not include batch norm
    :param act: The activation to apply after the conv op and batch norm (if included).
        Default is "relu", set to None for no activation.
    :param kernel_initializer: The initializer to use for the fully connected kernels
    :param bias_initializer: The initializer to use for the bias variable,
        if a bias is included
    :param beta_initializer: The initializer to use for the beta variable,
        if batch norm is included
    :param gamma_initializer: The initializer to use for the gamma variable,
        if gamma is included
    :return: the tensor after all ops have been applied
    """
    if include_bias is None:
        include_bias = not include_bn

    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        out = tf_compat.layers.dense(
            x_tens,
            units=channels,
            use_bias=include_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer if include_bias else None,
            name="fc",
        )

        if include_bn:
            out = tf_compat.layers.batch_normalization(
                out,
                axis=1,
                momentum=BN_MOMENTUM,
                epsilon=BN_EPSILON,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                training=training,
                name="bn",
            )

        if dropout_rate and dropout_rate > 0.0:
            out = tf_compat.layers.dropout(
                out, dropout_rate, training=training, name="dropout"
            )

        out = activation(out, act)

    return out


def fc(
    name: str,
    x_tens: tf_compat.Tensor,
    in_chan: int,
    out_chan: int,
    act: Union[None, str] = None,
):
    """
    Create a fully connected layer with the proper ops and variables.

    :param name: the name scope to create the layer under
    :param x_tens: the tensor to apply the layer to
    :param in_chan: the number of input channels
    :param out_chan: the number of output channels
    :param act: an activation type to add into the layer, supported:
        [None, relu, sigmoid, softmax]
    :return: the created layer
    """
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        weight = tf_compat.get_variable(
            "weight",
            shape=[in_chan, out_chan],
            initializer=tf_compat.glorot_normal_initializer(),
            dtype=tf_compat.float32,
        )
        bias = tf_compat.get_variable(
            "bias",
            shape=[out_chan],
            initializer=tf_compat.zeros_initializer(),
            dtype=tf_compat.float32,
        )

        x_tens = tf_compat.matmul(x_tens, weight, name="matmul")
        x_tens = tf_compat.nn.bias_add(x_tens, bias, name="bias_add")
        x_tens = activation(x_tens, act)

    return x_tens


def conv2d(
    name: str,
    x_tens: tf_compat.Tensor,
    in_chan: int,
    out_chan: int,
    kernel: int,
    stride: int,
    padding: str,
    act: Union[None, str] = None,
):
    """
    Create a convolutional layer with the proper ops and variables.

    :param name: the name scope to create the layer under
    :param x_tens: the tensor to apply the layer to
    :param in_chan: the number of input channels
    :param out_chan: the number of output channels
    :param kernel: the kernel size to create a convolution for
    :param stride: the stride to apply to the convolution
    :param padding: the padding to apply to the convolution
    :param act: an activation type to add into the layer, supported:
        [None, relu, sigmoid, softmax]
    :return: the created layer
    """
    with tf_compat.variable_scope(name, reuse=tf_compat.AUTO_REUSE):
        weight = tf_compat.get_variable(
            "weight",
            shape=[kernel, kernel, in_chan, out_chan],
            initializer=tf_compat.glorot_normal_initializer(),
            dtype=tf_compat.float32,
        )
        bias = tf_compat.get_variable(
            "bias",
            shape=[out_chan],
            initializer=tf_compat.zeros_initializer(),
            dtype=tf_compat.float32,
        )

        x_tens = tf_compat.nn.conv2d(
            x_tens, weight, strides=[1, stride, stride, 1], padding=padding, name="conv"
        )
        x_tens = tf_compat.nn.bias_add(x_tens, bias, name="bias_add")
        x_tens = activation(x_tens, act)

    return x_tens
