import os
from typing import Union

import pytest
import torch

from sparseml.pytorch.models import ModelRegistry, mobilenet
from tests.sparseml.pytorch.models.utils import compare_model


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_MODEL_TESTS", False),
    reason="Skipping model tests",
)
@pytest.mark.parametrize(
    "key,pretrained,test_input",
    [
        ("mobilenet", False, True),
        ("mobilenet", True, False),
        ("mobilenet", "base", False),
        ("mobilenet", "optim", False),
        ("mobilenet", "optim-perf", False),
    ],
)
def test_mobilenets(key: str, pretrained: Union[bool, str], test_input: bool):
    model = ModelRegistry.create(key, pretrained)
    diff_model = mobilenet()

    if pretrained:
        compare_model(model, diff_model, same=False)
        match_model = ModelRegistry.create(key, pretrained)
        compare_model(model, match_model, same=True)

    if test_input:
        input_shape = ModelRegistry.input_shape(key)
        batch = torch.randn(1, *input_shape)
        out = model(batch)
        assert isinstance(out, tuple)
        for tens in out:
            assert tens.shape[0] == 1
            assert tens.shape[1] == 1000
