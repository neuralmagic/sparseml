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
import torch

from sparseml.pytorch.models import (
    ModelRegistry,
    resnet18,
    resnet34,
    resnet50,
    resnet50_2xwidth,
    resnet101,
    resnet101_2xwidth,
    resnet152,
    resnetv2_18,
    resnetv2_34,
    resnetv2_50,
    resnetv2_101,
    resnetv2_152,
    resnext50,
    resnext101,
    resnext152,
)
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
    "key,pretrained,test_input,match_const",
    [
        ("resnet18", False, True, resnet18),
        ("resnet18", True, True, resnet18),
        ("resnet18", "base", True, resnet18),
        ("resnet18", "pruned-conservative", True, resnet18),
        ("resnetv2_18", False, True, resnetv2_18),
        ("resnet34", False, True, resnet34),
        ("resnet34", True, True, resnet34),
        ("resnet34", "base", True, resnet34),
        ("resnet34", "pruned-conservative", True, resnet34),
        ("resnetv2_34", False, True, resnetv2_34),
        ("resnet50", False, True, resnet50),
        ("resnet50", True, False, resnet50),
        ("resnet50", "base", False, resnet50),
        ("resnet50", "pruned-conservative", False, resnet50),
        ("resnet50", "pruned-moderate", False, resnet50),
        ("resnet50_2xwidth", False, True, resnet50_2xwidth),
        ("resnet50_2xwidth", True, False, resnet50_2xwidth),
        ("resnet50_2xwidth", "base", False, resnet50_2xwidth),
        ("resnetv2_50", False, True, resnetv2_50),
        ("resnext50", False, True, resnext50),
        ("resnet101", False, True, resnet101),
        ("resnet101", True, False, resnet101),
        ("resnet101", "base", False, resnet101),
        ("resnet101_2xwidth", False, True, resnet101_2xwidth),
        ("resnet101_2xwidth", True, False, resnet101_2xwidth),
        ("resnet101_2xwidth", "base", False, resnet101_2xwidth),
        ("resnetv2_101", False, True, resnetv2_101),
        ("resnext101", False, True, resnext101),
        ("resnet152", False, True, resnet152),
        ("resnet152", True, False, resnet152),
        ("resnet152", "base", False, resnet152),
        ("resnetv2_152", False, True, resnetv2_152),
        ("resnext152", False, True, resnext152),
    ],
)
def test_resnets(
    key: str, pretrained: Union[bool, str], test_input: bool, match_const: Callable
):
    model = ModelRegistry.create(key, pretrained)
    diff_model = match_const()

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
