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

import pytest
import torch
from torch import Tensor

from sparseml.pytorch.nn import SqueezeExcite


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
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
