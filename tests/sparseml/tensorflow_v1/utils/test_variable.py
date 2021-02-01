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
from typing import List

import pytest

from sparseml.tensorflow_v1.utils import (
    clean_tensor_name,
    get_op_input_var,
    get_ops_and_inputs_by_name_or_regex,
    get_prunable_ops,
    tf_compat,
)
from tests.sparseml.tensorflow_v1.helpers import conv_net, mlp_net


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_op_var_name():
    graph = tf_compat.Graph()

    with graph.as_default():
        var = tf_compat.Variable(
            tf_compat.random_normal([64]), dtype=tf_compat.float32, name="test_var_name"
        )
        name = clean_tensor_name(var)
        assert name == "test_var_name"


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
def test_op_input_var():
    with tf_compat.Graph().as_default() as graph:
        mlp_net()
        ops = get_prunable_ops(graph)

        for op in ops:
            inp = get_op_input_var(op[1])
            assert inp is not None
            assert isinstance(inp, tf_compat.Tensor)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_TENSORFLOW_TESTS", False),
    reason="Skipping tensorflow_v1 tests",
)
@pytest.mark.parametrize(
    "net_const,expected_ops",
    [
        (mlp_net, ["mlp_net/fc1/matmul", "mlp_net/fc2/matmul", "mlp_net/fc3/matmul"]),
        (
            conv_net,
            ["conv_net/conv1/conv", "conv_net/conv2/conv", "conv_net/mlp/matmul"],
        ),
    ],
)
def test_get_prunable_ops(net_const, expected_ops: List[str]):
    with tf_compat.Graph().as_default():
        net_const()
        ops = get_prunable_ops()
        assert len(ops) == len(expected_ops)

        for op in ops:
            assert op[0] in expected_ops


@pytest.mark.parametrize(
    "net_const,var_names,expected_ops,expected_tens",
    [
        (
            mlp_net,
            ["mlp_net/fc1/weight", "mlp_net/fc2/weight", "mlp_net/fc3/weight"],
            ["mlp_net/fc1/matmul", "mlp_net/fc2/matmul", "mlp_net/fc3/matmul"],
            ["mlp_net/fc1/weight", "mlp_net/fc2/weight", "mlp_net/fc3/weight"],
        ),
        (
            mlp_net,
            ["mlp_net/fc1/weight"],
            ["mlp_net/fc1/matmul"],
            ["mlp_net/fc1/weight"],
        ),
        (
            conv_net,
            ["re:conv_net/.*/weight"],
            ["conv_net/conv1/conv", "conv_net/conv2/conv", "conv_net/mlp/matmul"],
            ["conv_net/conv1/weight", "conv_net/conv2/weight", "conv_net/mlp/weight"],
        ),
    ],
)
def test_get_ops_and_inputs_by_name_or_regex(
    net_const,
    var_names,
    expected_ops,
    expected_tens,
):
    with tf_compat.Graph().as_default() as graph:
        net_const()
        ops_and_inputs = get_ops_and_inputs_by_name_or_regex(var_names, graph)
        assert len(ops_and_inputs) == len(expected_ops)

        for op, inp in ops_and_inputs:
            assert op.name in expected_ops
            assert clean_tensor_name(inp.name) in expected_tens
