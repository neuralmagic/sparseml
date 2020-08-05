import pytest

import numpy
import os

from neuralmagicML.onnx.recal import prune_unstructured


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "array,sparsities",
    [
        (
            numpy.random.randn(3, 128, 128),
            [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 0.99, 0.999],
        ),
    ],
)
def test_prune_unstructured(array, sparsities):

    for sparsity in sparsities:
        pruned_array = prune_unstructured(array, sparsity)
        measured_sparsity = float(
            pruned_array.size - numpy.count_nonzero(pruned_array)
        ) / float(pruned_array.size)
        assert abs(measured_sparsity - sparsity) < 1e-4
