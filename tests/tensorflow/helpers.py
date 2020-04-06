from neuralmagicML.tensorflow.utils import tf_compat


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
    graph = tf_compat.Graph()

    with graph.as_default():
        with tf_compat.name_scope("mlp_net"):
            inp = tf_compat.placeholder(tf_compat.float32, [None, 8], name="inp")
            fc1 = _fc("fc1", inp, 8, 16)
            fc2 = _fc("fc2", fc1, 16, 32)
            fc3 = _fc("fc3", fc2, 32, 64, add_relu=False)
            out = tf_compat.sigmoid(fc3, name="sig")

    return graph


def conv_net():
    graph = tf_compat.Graph()

    with graph.as_default():
        with tf_compat.name_scope("conv_net"):
            inp = tf_compat.placeholder(tf_compat.float32, [None, 28, 28, 1])
            conv1 = _conv("conv1", inp, 1, 16, 3, 2, "SAME")
            conv2 = _conv("conv2", conv1, 16, 32, 3, 2, "SAME")
            avg_pool = tf_compat.reduce_mean(conv2, axis=[1, 2])
            reshape = tf_compat.reshape(avg_pool, [-1, 32])
            mlp = _fc("mlp", reshape, 32, 10, add_relu=False)
            sig = tf_compat.sigmoid(mlp)

    return graph
