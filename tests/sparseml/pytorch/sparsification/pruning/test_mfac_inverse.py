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

from flaky import flaky
from sparseml.pytorch.sparsification.pruning import (
    EmpiricalBlockFisherInverse,
    FisherInverseFast,
    FisherInverseFastBlock,
    FisherInverseFastSmallBlocks,
)


# precent-wise precision in terms of the mean of the resulting tensors
PRECISION = 0.00001


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "fisher_algorithm",
    [
        FisherInverseFastSmallBlocks,
        FisherInverseFastBlock,
    ],
)
@pytest.mark.parametrize(
    "devices",
    [
        None,
        pytest.param(
            ["cuda:0"],
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="No CUDA devices available",
            ),
        ),
    ],
)
@flaky(max_runs=3, min_passes=2)
def test_blocked_fisher_inverse(fisher_algorithm, devices):
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

    blocked_hinv = fisher_algorithm(
        grads=grads2,
        block_size=block_size,
        damp=damp,
        devices=devices,
    )

    fast_diag = fast_hinv.diag()
    small_blocks_diag = blocked_hinv.diag()

    tensor_to_mul = torch.rand(total_params)
    fast_mul_out = fast_hinv.mul(tensor_to_mul)
    small_blocks_mul_out = blocked_hinv.mul(tensor_to_mul)

    assert (
        pytest.approx(
            fast_diag,
            torch.sqrt(torch.mean(torch.square(fast_diag))).item() * PRECISION,
        )
        == small_blocks_diag
    )
    assert (
        pytest.approx(
            fast_mul_out,
            torch.sqrt(torch.mean(torch.square(fast_mul_out))).item() * PRECISION,
        )
        == small_blocks_mul_out
    )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "fisher_block_online_algorithm",
    [
        EmpiricalBlockFisherInverse,
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="No CUDA devices available",
            ),
        ),
    ],
)
@flaky(max_runs=3, min_passes=2)
def test_online_blocked_fisher_inverse(fisher_block_online_algorithm, device):
    total_params = 10000
    num_grads = 32
    block_size = 1000
    damp = 0.000001

    grads1 = torch.rand(num_grads, total_params)
    grads2 = grads1.clone()

    finv_smallblocks = FisherInverseFastSmallBlocks(
        grads=grads1,
        block_size=block_size,
        damp=damp,
        devices=[device],
    )

    finv_block_online = fisher_block_online_algorithm(
        num_grads=num_grads,
        fisher_block_size=block_size,
        num_weights=total_params,
        damp=damp,
        device=device,
    )
    for i in range(num_grads):
        finv_block_online.add_grad(grads2[i, :].to(device))

    finv_diag = finv_smallblocks.diag()
    finv_online_diag = finv_block_online.diag().cpu()

    tensor_to_mul = torch.rand(total_params, device=device)
    finv_mul_out = finv_smallblocks.mul(tensor_to_mul)
    finv_online_mul_out = finv_block_online.mul(tensor_to_mul).cpu()

    assert (
        pytest.approx(
            finv_diag,
            torch.sqrt(torch.mean(torch.square(finv_diag))).item() * PRECISION,
        )
        == finv_online_diag
    )
    assert (
        pytest.approx(
            finv_mul_out,
            torch.sqrt(torch.mean(torch.square(finv_mul_out))).item() * PRECISION,
        )
        == finv_online_mul_out
    )
