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
import sys

import numpy
import pytest
import torch

from sparseml.pytorch.utils import (
    MFACOptions,
    FisherInverseFast, 
    FisherInverseFastBlock, 
    FisherInverseFastSmallBlocks,
)

@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fast_small_blocks():
    total_params = 1000
    block_size = 1000
    num_grads = 32
    damp = .0000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()
    fast_hinv = FisherInverseFast(
        grads = grads1,
        damp= damp,
    )

    small_blocks_hinv = FisherInverseFastSmallBlocks(
        grads = grads2,
        block_size = block_size,
        damp= damp,
    )

    fast_diag = fast_hinv.diag()
    small_blocks_diag = small_blocks_hinv.diag()
    assert pytest.approx(small_blocks_diag) == fast_diag

@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fast_large_blocks():
    total_params = 1000
    block_size = 1000
    num_grads = 32
    damp = .0000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()
    fast_hinv = FisherInverseFast(
        grads = grads1,
        damp= 1.0 / damp,
    )

    large_blocks_hinv = FisherInverseFastBlock(
        grads = grads2,
        block_size = block_size,
        damp=1.0 / damp,
    )

    fast_diag = fast_hinv.diag()
    large_blocks_diag = large_blocks_hinv.diag()
    assert pytest.approx(large_blocks_diag) == fast_diag