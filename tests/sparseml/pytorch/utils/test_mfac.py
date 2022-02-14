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

from sparseml.pytorch.utils import (
    FisherInverseFast,
    FisherInverseFastBlock,
    FisherInverseFastSmallBlocks,
)


# precent-wise precision in terms of the mean of the resulting tensors
PRECISION = 0.000001


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fast_small_blocks():
    total_params = 1000
    num_grads = 32
    block_size = total_params
    damp = 0.0000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()

    fast_hinv = FisherInverseFast(
        grads=grads1,
        damp=damp,
    )

    small_blocks_hinv = FisherInverseFastSmallBlocks(
        grads=grads2,
        block_size=block_size,
        damp=damp,
    )

    fast_diag = fast_hinv.diag()
    small_blocks_diag = small_blocks_hinv.diag()

    tensor_to_mul = torch.rand(total_params)
    fast_mul_out = fast_hinv.mul(tensor_to_mul)
    small_blocks_mul_out = small_blocks_hinv.mul(tensor_to_mul)

    assert (
        pytest.approx(fast_diag, torch.mean(fast_diag).item() * PRECISION)
        == small_blocks_diag
    )
    assert (
        pytest.approx(
            fast_mul_out,
            torch.mean(fast_mul_out).item() * PRECISION,
        )
        == small_blocks_mul_out
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False) or (torch.cuda.device_count() == 0),
    reason="Skipping pytorch tests",
)
def test_fast_small_blocks_gpu():
    total_params = 1000
    num_grads = 32
    block_size = total_params
    damp = 0.0000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()

    fast_hinv = FisherInverseFast(
        grads=grads1,
        damp=damp,
    )

    small_blocks_hinv = FisherInverseFastSmallBlocks(
        grads=grads2, block_size=block_size, damp=damp, devices=["cuda:0"]
    )

    fast_diag = fast_hinv.diag()
    small_blocks_diag = small_blocks_hinv.diag()

    tensor_to_mul = torch.rand(total_params)
    fast_mul_out = fast_hinv.mul(tensor_to_mul)
    small_blocks_mul_out = small_blocks_hinv.mul(tensor_to_mul)

    assert (
        pytest.approx(fast_diag, torch.mean(fast_diag).item() * PRECISION)
        == small_blocks_diag
    )
    assert (
        pytest.approx(
            fast_mul_out,
            torch.mean(fast_mul_out).item() * PRECISION,
        )
        == small_blocks_mul_out
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_fast_large_blocks():
    total_params = 1000
    num_grads = 32
    block_size = total_params
    damp = 0.0000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()

    fast_hinv = FisherInverseFast(
        grads=grads1,
        damp=damp,
    )

    small_blocks_hinv = FisherInverseFastBlock(
        grads=grads2,
        block_size=block_size,
        damp=damp,
    )

    fast_diag = fast_hinv.diag()
    small_blocks_diag = small_blocks_hinv.diag()

    tensor_to_mul = torch.rand(total_params)
    fast_mul_out = fast_hinv.mul(tensor_to_mul)
    small_blocks_mul_out = small_blocks_hinv.mul(tensor_to_mul)

    assert (
        pytest.approx(fast_diag, torch.mean(fast_diag).item() * PRECISION)
        == small_blocks_diag
    )
    assert (
        pytest.approx(
            fast_mul_out,
            torch.mean(fast_mul_out).item() * PRECISION,
        )
        == small_blocks_mul_out
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False) or (torch.cuda.device_count() == 0),
    reason="Skipping pytorch tests",
)
def test_fast_large_blocks_gpu():
    total_params = 1000
    num_grads = 32
    block_size = total_params
    damp = 0.0000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()

    fast_hinv = FisherInverseFast(
        grads=grads1,
        damp=damp,
    )

    small_blocks_hinv = FisherInverseFastBlock(
        grads=grads2, block_size=block_size, damp=damp, devices=["cuda:0"]
    )

    fast_diag = fast_hinv.diag()
    small_blocks_diag = small_blocks_hinv.diag()

    tensor_to_mul = torch.rand(total_params)
    fast_mul_out = fast_hinv.mul(tensor_to_mul)
    small_blocks_mul_out = small_blocks_hinv.mul(tensor_to_mul)

    assert (
        pytest.approx(fast_diag, torch.mean(fast_diag).item() * PRECISION)
        == small_blocks_diag
    )
    assert (
        pytest.approx(
            fast_mul_out,
            torch.mean(fast_mul_out).item() * PRECISION,
        )
        == small_blocks_mul_out
    )
