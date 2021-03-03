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

import os
from typing import Callable

import numpy
import pytest

from sparseml.tensorflow_v1.optim.sensitivity_pruning import (
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
    pruning_loss_sens_op_vars,
)
from sparseml.tensorflow_v1.utils import batch_cross_entropy_loss, tf_compat
from tests.sparseml.tensorflow_v1.helpers import mlp_net


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "net_const",
    [mlp_net],
)
def test_approx_ks_loss_sensitivity(net_const: Callable):
    with tf_compat.Graph().as_default() as graph:
        out, inp = net_const()

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())

            analysis = pruning_loss_sens_magnitude(graph)

            for res in analysis.results:
                assert res.name
                assert isinstance(res.index, int)
                assert len(res.sparse_measurements) > 0
                assert len(res.averages) > 0
                assert res.sparse_average > 0
                assert res.sparse_integral > 0


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "net_const,inp_arr,labs_arr",
    [(mlp_net, numpy.random.random((8, 16)), numpy.random.random((8, 64)))],
)
def test_loss_sensitivity(
    net_const: Callable, inp_arr: numpy.ndarray, labs_arr: numpy.ndarray
):
    with tf_compat.Graph().as_default():
        out, inp = net_const()
        labels = tf_compat.placeholder(
            tf_compat.float32, [None, *labs_arr.shape[1:]], name="logits"
        )
        loss = batch_cross_entropy_loss(out, labels)
        op_vars = pruning_loss_sens_op_vars()

        with tf_compat.Session() as sess:
            sess.run(tf_compat.global_variables_initializer())

            def add_ops_creator(step: int):
                return []

            def feed_dict_creator(step: int):
                return {inp: inp_arr, labels: labs_arr}

            analysis = pruning_loss_sens_one_shot(
                op_vars, loss, 5, add_ops_creator, feed_dict_creator
            )

            for res in analysis.results:
                assert res.name
                assert isinstance(res.index, int)
                assert len(res.sparse_measurements) > 0
                assert len(res.averages) > 0
                assert res.sparse_average > 0
                assert res.sparse_integral > 0
