import os
import pytest

from typing import Union, Callable
import torch
from sparseml.pytorch.models import ModelRegistry, darknet53

from tests.pytorch.models.utils import compare_model


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_MODEL_TESTS", False), reason="Skipping model tests",
)
@pytest.mark.parametrize(
    "key,pretrained,test_input,match_const, match_args",
    [("darknet_53", False, True, darknet53, {})],
)
def test_darknet(
    key: str,
    pretrained: Union[bool, str],
    match_const: Callable,
    test_input: bool,
    match_args: dict,
):
    model = ModelRegistry.create(key, pretrained)
    diff_model = match_const(**match_args)

    if pretrained:
        compare_model(model, diff_model, same=False)
        match_model = ModelRegistry.create(key, pretrained)
        compare_model(model, match_model, same=True)

    if test_input:
        input_shape = ModelRegistry.input_shape(key)
        batch = torch.randn(1, *input_shape)
        model = model.eval()
        out = model(batch)
        assert isinstance(out, tuple)
        for tens in out:
            assert tens.shape[0] == 1
            assert tens.shape[1] == 1000
