import pytest
import numpy as np
import tensorflow as tf


from neuralmagicML.tensorflow.recal.kernel import one_shot_ks_loss_sensitivity
from neuralmagicML.tensorflow.utils import tf_compat

RANDOM_SEED = 2020
np.random.seed(RANDOM_SEED)


def simple_matmul_net():
    tf_compat.reset_default_graph()
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    X = tf_compat.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf_compat.placeholder(tf.int64, shape=(None,), name="y")

    def neuron_layer(X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b
            if activation is not None:
                return activation(Z)
            else:
                return Z

    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = neuron_layer(
            hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu
        )
        logits = neuron_layer(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        loss = tf.reduce_mean(xentropy, name="loss")

    return tf_compat.get_default_graph(), X, y, loss


@pytest.mark.parametrize(
    "model_func, X, y, batch_size, samples_per_measurement",
    [
        (
            simple_matmul_net,
            np.random.rand(1024, 28 * 28),
            np.random.randint(0, high=10, size=1024),
            8,
            256,
        )
    ],
)
def test_explicit_loss(
    model_func, X: np.array, y: np.array, batch_size, samples_per_measurement
):
    graph, X_placeholder, y_placeholder, total_loss = model_func()
    with tf_compat.Session(graph=graph) as sess:
        sess.run(tf_compat.global_variables_initializer())
        analysis = one_shot_ks_loss_sensitivity(
            sess,
            graph,
            X,
            y,
            X_placeholder,
            y_placeholder,
            total_loss,
            batch_size,
            samples_per_measurement,
        )
        for res in analysis.results:
            assert res.integral > 0.0
