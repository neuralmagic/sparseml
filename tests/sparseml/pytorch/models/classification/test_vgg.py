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
    vgg11,
    vgg11bn,
    vgg13,
    vgg13bn,
    vgg16,
    vgg16bn,
    vgg19,
    vgg19bn,
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
        ("vgg11", False, True, vgg11),
        ("vgg11", True, False, vgg11),
        ("vgg11bn", False, True, vgg11bn),
        ("vgg11bn", True, False, vgg11bn),
        ("vgg13", False, True, vgg13),
        ("vgg13", True, False, vgg13),
        ("vgg13bn", False, True, vgg13bn),
        ("vgg13bn", True, False, vgg13bn),
        ("vgg16", False, True, vgg16),
        ("vgg16", True, False, vgg16),
        ("vgg16", "base", False, vgg16),
        ("vgg16", "pruned-conservative", False, vgg16),
        ("vgg16", "pruned-moderate", False, vgg16),
        ("vgg16bn", False, True, vgg16bn),
        ("vgg16bn", True, False, vgg16bn),
        ("vgg19", False, True, vgg19),
        ("vgg19", True, False, vgg19),
        ("vgg19bn", False, True, vgg19bn),
        ("vgg19bn", True, False, vgg19bn),
    ],
)
def test_vggs(
    key: str, pretrained: Union[bool, str], match_const: Callable, test_input: bool
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
