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
from typing import Callable, Union

import pytest
from torchvision import models as torchvision_models

from sparseml.pytorch.models import ModelRegistry
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
