import pytest

from typing import Callable
import numpy

from neuralmagicML.tensorflow.recal.sensitivity_ks import (
    ks_loss_sensitivity_op_vars,
    one_shot_ks_loss_sensitivity,
)
from neuralmagicML.tensorflow.utils import tf_compat, batch_cross_entropy_loss

from tests.tensorflow.helpers import mlp_net


@pytest.mark.parametrize(
    "net_const,inp_arr,labs_arr",
    [(mlp_net, numpy.random.random((8, 16)), numpy.random.random((8, 64)))],
)
def test_loss_sensitivity(
    net_const: Callable, inp_arr: numpy.ndarray, labs_arr: numpy.ndarray
):
    with tf_compat.Graph().as_default() as graph:
        out, inp = net_const()
        labels = tf_compat.placeholder(
            tf_compat.float32, [None, *labs_arr.shape[1:]], name="logits"
        )
        loss = batch_cross_entropy_loss(out, labels)
        op_vars = ks_loss_sensitivity_op_vars()

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())

            def add_ops_creator(step: int):
                return []

            def feed_dict_creator(step: int):
                return {inp: inp_arr, labels: labs_arr}

            sens = one_shot_ks_loss_sensitivity(
                op_vars, loss, 5, add_ops_creator, feed_dict_creator
            )
