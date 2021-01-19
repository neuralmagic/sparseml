import os
from typing import Callable, Union

import pytest
import torch
from torchvision import models as torchvision_models

from sparseml.pytorch.models import ModelRegistry
from tests.pytorch.models.utils import compare_model


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_MODEL_TESTS", False),
    reason="Skipping model tests",
)
@pytest.mark.parametrize(
    "key,pretrained,constructor",
    [
        ("torchvision.mobilenet_v2", False, torchvision_models.mobilenet_v2),
        ("torchvision.mobilenet_v2", True, torchvision_models.mobilenet_v2),
        ("torchvision.resnet50", False, torchvision_models.resnet50),
        ("torchvision.resnet50", True, torchvision_models.resnet50),
    ],
)
def test_torchvision_registry_models(
    key: str, pretrained: Union[bool, str], constructor: Callable
):
    model = ModelRegistry.create(key, pretrained)
    diff_model = constructor(pretrained=False)
    compare_model(model, diff_model, same=False)

    if pretrained is True:
        # check torchvision weights are properly loaded
        match_model = constructor(pretrained=pretrained)
        compare_model(model, match_model, same=True)
