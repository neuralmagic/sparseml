import pytest

from typing import List

from neuralmagicML.tensorflow.utils import get_prunable_ops

from tests.tensorflow.helpers import mlp_net, conv_net


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
