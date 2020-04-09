from typing import Union

from neuralmagicML.tensorflow.utils import tf_compat


__all__ = ["activation", "fc", "conv2d"]


def activation(x_tens: tf_compat.Tensor, act: Union[None, str]):
    """
    Create an activation operation.

    :param x_tens: the tensor to apply the op to
    :param act: the activation type to apply, supported: [None, relu, sigmoid, softmax]
    :return: the created operation
    """
    if not act:
        return x_tens

    if act == "relu":
        return tf_compat.nn.relu(x_tens, name="act")

    if act == "sigmoid":
        return tf_compat.nn.sigmoid(x_tens, name="act")

    if act == "softmax":
        return tf_compat.nn.softmax(x_tens, name="act")

    raise ValueError("unknown act given of {}".format(act))


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
