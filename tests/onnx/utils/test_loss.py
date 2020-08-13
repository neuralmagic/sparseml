import pytest
import numpy

from neuralmagicML.onnx.utils.loss import kl_divergence


@pytest.mark.parametrize(
    "distribution, other_distribution",
    [
        (numpy.random.rand(3, 120, 120), numpy.random.rand(3, 120, 120)),
        (numpy.random.randn(3, 120, 120), numpy.random.randn(3, 120, 120)),
        (numpy.random.randn(3, 120, 120), numpy.random.rand(3, 120, 120)),
        (numpy.random.randn(3, 120, 120), numpy.ones((3, 120, 120)) + 1),
    ],
)
def test_kl_divergence(distribution: numpy.array, other_distribution: numpy.array):
    previous_kl_div = None
    zero_point_and_min_value_pairs = [
        (2, 1),
        (1, 1),
        (0, 2),
        (0, 1),
        (0, 0.25),
        (0, 0.01),
        (-1, -1),
    ]
    for zero_point, min_value in zero_point_and_min_value_pairs:
        outs = []
        for _ in range(100):
            outs.append(
                kl_divergence(
                    distribution,
                    other_distribution,
                    zero_point=zero_point,
                    min_value=min_value,
                )
            )

        if previous_kl_div is None:
            previous_kl_div = numpy.mean(outs)
        else:
            assert previous_kl_div <= numpy.mean(outs)
            previous_kl_div = numpy.mean(outs)
