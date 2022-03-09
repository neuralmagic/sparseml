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

from sparseml.pytorch.models import ModelRegistry, yolo_v3
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
    "key,pretrained,pretrained_backbone,test_input,match_const",
    [("yolo_v3", False, False, True, yolo_v3)],
)
def test_yolo_v3(
    key: str,
    pretrained: Union[bool, str],
    pretrained_backbone: Union[bool, str],
    test_input: bool,
    match_const: Callable,
):
    model = ModelRegistry.create(key, pretrained)
    diff_model = match_const(pretrained_backbone=pretrained_backbone)

    if pretrained:
        compare_model(model, diff_model, same=False)
        match_model = ModelRegistry.create(key, pretrained)
        compare_model(model, match_model, same=True)

    if pretrained_backbone and pretrained_backbone is not True:
        compare_model(model.backbone, diff_model.backbone, same=False)
        match_model = ModelRegistry.create(key, pretrained_backbone=pretrained_backbone)
        compare_model(diff_model.backbone, match_model.backbone, same=True)

    if test_input:
        input_shape = ModelRegistry.input_shape(key)
        batch = torch.randn(1, *input_shape)
        model.eval()
        outputs = model(batch)
        assert isinstance(outputs, list)
        for output in outputs:
            assert isinstance(output, torch.Tensor)
            assert output.dim() == 5
            assert output.size(-1) == 85
