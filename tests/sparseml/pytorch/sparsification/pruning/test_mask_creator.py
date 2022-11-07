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

import pytest
import torch

from sparseml.pytorch.sparsification.pruning import (
    BlockMaskCreator,
    FourBlockMaskCreator,
    GroupedPruningMaskCreator,
    NMPruningMaskCreator,
    UnstructuredPruningMaskCreator,
)
from sparseml.pytorch.utils import tensor_sparsity
from tests.sparseml.pytorch.sparsification.pruning.helpers import (
    grouped_masks_test,
    sparsity_mask_creator_test,
)


@pytest.mark.parametrize(
    ("tensor_shape,mask_creator"),
    [
        ([[64, 64]] * 10, UnstructuredPruningMaskCreator()),
        ([[64, 64, 3, 3], [64, 64]], UnstructuredPruningMaskCreator()),
        ([[64, 513]], FourBlockMaskCreator()),
        ([[64, 512, 3, 3]], FourBlockMaskCreator()),
        ([[63, 513, 3, 3]], FourBlockMaskCreator()),
        ([[64, 512, 3, 3]], BlockMaskCreator([1, 4])),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val):
    sparsity_mask_creator_test(tensor_shape, mask_creator, sparsity_val, "cpu")


@pytest.mark.parametrize(
    ("tensor_shape,mask_creator"),
    [
        ([[64, 64, 3, 3]], UnstructuredPruningMaskCreator()),
        ([[64, 512, 3, 3]], FourBlockMaskCreator()),
        ([[64, 512, 3, 3]], BlockMaskCreator([1, 4])),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_sparsity_mask_creator_cuda(tensor_shape, mask_creator, sparsity_val):
    sparsity_mask_creator_test(tensor_shape, mask_creator, sparsity_val, "cuda")


@pytest.mark.parametrize(
    ("tensors,mask_creator"),
    [
        (
            [torch.randn(128, 128, 3, 3), 3 * torch.randn(64, 64)],
            UnstructuredPruningMaskCreator(),
        ),
        (
            [i * torch.randn(64, 64, 3, 3) for i in range(1, 6)],
            UnstructuredPruningMaskCreator(),
        ),
        (
            [torch.randn(128, 128, 3, 3), 3 * torch.randn(64, 512)],
            FourBlockMaskCreator(),
        ),
        (
            [i * torch.randn(64, 64, 3, 3) for i in range(1, 6)],
            FourBlockMaskCreator(),
        ),
        (
            [torch.randn(128, 128, 3, 3), 3 * torch.randn(64, 512)],
            BlockMaskCreator([1, 4]),
        ),
        (
            [i * torch.randn(64, 64, 3, 3) for i in range(1, 6)],
            BlockMaskCreator([1, 4]),
        ),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_global_sparsity_mask_creator(tensors, mask_creator, sparsity_val):
    masks = mask_creator.create_sparsity_masks(
        tensors, sparsity_val, global_sparsity=True
    )
    mask_sparsities = [tensor_sparsity(mask) for mask in masks]
    global_sparsity = tensor_sparsity(torch.cat([mask.reshape(-1) for mask in masks]))
    assert abs(global_sparsity - sparsity_val) < 1e-2

    if sparsity_val not in [0.0, 1.0]:
        # check that individual sparsity masks are reasonably dissimilar
        assert len(set(mask_sparsities)) > 1

    if isinstance(mask_creator, GroupedPruningMaskCreator):
        grouped_masks_test(masks, mask_creator)


@pytest.mark.parametrize(
    ("tensor_shapes,mask_creator,sparsity_val"),
    [
        (
            [[128, 128, 3, 3], [64, 64]],
            UnstructuredPruningMaskCreator(),
            [0.8, 0.9],
        ),
        (
            [[64, 64, 3, 3]] * 6,
            UnstructuredPruningMaskCreator(),
            [0.4, 0.6, 0.8, 0.9, 0.95, 0.99],
        ),
        (
            [[128, 128, 3, 3], [64, 512]],
            FourBlockMaskCreator(),
            [0.8, 0.9],
        ),
        (
            [[64, 64, 3, 3]] * 6,
            FourBlockMaskCreator(),
            [0.4, 0.6, 0.8, 0.9, 0.95, 0.99],
        ),
        (
            [[128, 128, 3, 3], [64, 512]],
            BlockMaskCreator([1, 4]),
            [0.8, 0.9],
        ),
        (
            [[64, 64, 3, 3]] * 6,
            BlockMaskCreator([1, 4]),
            [0.4, 0.6, 0.8, 0.9, 0.95, 0.99],
        ),
    ],
)
def test_sparsity_mask_creator_mult_tensor(tensor_shapes, mask_creator, sparsity_val):
    sparsity_mask_creator_test(tensor_shapes, mask_creator, sparsity_val, "cpu")


@pytest.mark.parametrize(
    ("tensors"),
    [
        [torch.randn(128, 128), torch.randn(128, 512, 3, 3)],
        [torch.randn(5, 64, 3, 3)],
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_four_block_mask_creator_matches_block(tensors, sparsity_val):
    mask_creator_1 = FourBlockMaskCreator()
    mask_creator_2 = BlockMaskCreator([1, 4])

    masks_1 = mask_creator_1.create_sparsity_masks(tensors, sparsity_val)
    masks_2 = mask_creator_2.create_sparsity_masks(tensors, sparsity_val)

    for mask_1, mask_2 in zip(masks_1, masks_2):
        assert mask_1.shape == mask_2.shape
        assert torch.all(mask_1 == mask_2)


@pytest.mark.parametrize(
    "N, M",
    [(2, 4), (3, 4), (1, 8), (7, 8)],
    scope="function",
)
@pytest.mark.parametrize(
    "tensor_shape",
    [[[64, 64]] * 10, [[64, 64, 3, 3]], [[64, 513]]],
)
class TestNMPruningMaskCreator:
    def test_sparsity_mask_creator(self, N, M, tensor_shape):
        sparsity_val = 1 - (N / M)
        sparsity_mask_creator_test(
            tensor_shape, NMPruningMaskCreator(N, M), sparsity_val, "cpu"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires cuda availability"
    )
    def test_sparsity_mask_creator_cuda(self, N, M, tensor_shape):
        sparsity_val = 1 - (N / M)
        sparsity_mask_creator_test(
            tensor_shape, NMPruningMaskCreator(N, M), sparsity_val, "cuda"
        )

    def test_pre_prune_sparsity(self, N, M, tensor_shape):
        sparsity_val = 0.0
        sparsity_mask_creator_test(
            tensor_shape, NMPruningMaskCreator(N, M), sparsity_val, "cpu"
        )

    def test_mask_structure_pattern(self, N, M, tensor_shape):
        sparsity_val = 1 - (N / M)
        mask_creator = NMPruningMaskCreator(N, M)
        tensors = [torch.randn(shape) for shape in tensor_shape]
        masks = mask_creator.create_sparsity_masks(tensors, sparsity_val)

        for mask in masks:
            flat_mask = torch.flatten(mask)
            for i in range(torch.numel(flat_mask) // M):
                assert torch.sum(flat_mask[i * M : (i + 1) * M]) == N
