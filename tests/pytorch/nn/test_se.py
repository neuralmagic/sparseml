import pytest

import os
import torch
from torch import Tensor

from neuralmagicML.pytorch.nn import SqueezeExcite


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_squeeze_excite():
    channels = 64
    x_tens = torch.randn(1, channels, 24, 24)
    se = SqueezeExcite(channels, 16)
    out = se(x_tens)  # type: Tensor

    assert out.shape[0] == 1
    assert out.shape[1] == channels
    assert out.shape[2] == 1
    assert out.shape[3] == 1
    assert len(out[out < 0.0]) == 0
    assert len(out[out > 1.0]) == 0
