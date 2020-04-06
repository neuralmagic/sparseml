import pytest

from typing import List

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    get_var_name,
    get_op_input_var,
    get_prunable_ops,
)

from tests.tensorflow.helpers import mlp_net, conv_net


def test_op_var_name():
    graph = tf_compat.Graph()

    with graph.as_default():
        var = tf_compat.Variable(
            tf_compat.random_normal([64]), dtype=tf_compat.float32, name="test_var_name"
        )
        name = get_var_name(var)
        assert name == "test_var_name:0"


def test_op_input_var():
    graph = mlp_net()
    ops = get_prunable_ops(graph)

    for op in ops:
        inp = get_op_input_var(op[1])
        assert inp is not None
        assert isinstance(inp, tf_compat.Tensor)


@pytest.mark.parametrize(
    "graph,expected_ops",
    [
        (mlp_net(), ["mlp_net/fc1/matmul", "mlp_net/fc2/matmul", "mlp_net/fc3/matmul"]),
        (
            conv_net(),
            ["conv_net/conv1/conv", "conv_net/conv2/conv", "conv_net/mlp/matmul"],
        ),
    ],
)
def test_get_prunable_ops(graph, expected_ops: List[str]):
    ops = get_prunable_ops(graph)
    assert len(ops) == len(expected_ops)

    for op in ops:
        assert op[0] in expected_ops
