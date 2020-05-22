import os
import pytest

from typing import Union
import torch

from neuralmagicML.pytorch.models import ModelRegistry, mnist_net

from tests.pytorch.models.utils import compare_model


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_MODEL_TESTS", False), reason="Skipping model tests",
)
@pytest.mark.parametrize(
    "key,pretrained,test_input",
    [
        ("mnistnet", False, True),
        ("mnistnet", True, False),
        ("mnistnet", "base", False),
    ],
)
def test_mnist(key: str, pretrained: Union[bool, str], test_input: bool):
    model = ModelRegistry.create(key, pretrained)
    diff_model = mnist_net()

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
            assert tens.shape[1] == 10
