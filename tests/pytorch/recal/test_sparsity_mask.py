import pytest
import torch

from neuralmagicML.pytorch.recal import (
    UnstructuredSparsityMaskCreator,
    GroupedSparsityMaskCreator,
    DimensionSparsityMaskCreator,
    BlockSparsityMaskCreator,
    load_mask_creator,
)
from neuralmagicML.pytorch.utils import tensor_sparsity


def _test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val, device):
    tensor = torch.randn(tensor_shape)
    tensor.to(device)
    initial_mask = mask_creator.create_sparsity_mask_from_tensor(tensor)
    update_mask = mask_creator.create_sparsity_mask(tensor, sparsity_val)

    update_mask_sparsity = tensor_sparsity(update_mask)
    assert abs(update_mask_sparsity - sparsity_val) < 1e-2
    if isinstance(mask_creator, GroupedSparsityMaskCreator):
        # Check that every value in the mask_creator grouping
        # is the same within the mask.  Assumes grouping applies
        # an absolte mean to each grouping
        for mask in [initial_mask, update_mask]:
            grouped_mask = mask_creator._group_tensor(mask)
            mask_vals_are_grouped = torch.all(
                torch.logical_or(grouped_mask == 0.0, grouped_mask == 1.0)
            )
            assert mask_vals_are_grouped


@pytest.mark.parametrize(
    ("tensor_shape,mask_creator"),
    [
        ([64, 64], UnstructuredSparsityMaskCreator()),
        ([64, 64, 3, 3], UnstructuredSparsityMaskCreator()),
        ([64, 512], DimensionSparsityMaskCreator(1)),
        ([64, 512, 3, 3], DimensionSparsityMaskCreator(1)),
        ([64, 64, 3, 3], DimensionSparsityMaskCreator([0, 1])),
        ([64, 512], BlockSparsityMaskCreator([1, 4])),
        ([64, 512, 3, 3], BlockSparsityMaskCreator([1, 4])),
        ([128, 128, 3, 3], BlockSparsityMaskCreator([2, 2])),
        ([128, 128, 3, 3], BlockSparsityMaskCreator([-1, 1])),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val):
    _test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val, "cpu")


@pytest.mark.parametrize(
    ("tensor_shape,mask_creator"),
    [
        ([64, 64, 3, 3], UnstructuredSparsityMaskCreator()),
        ([64, 64, 3, 3], DimensionSparsityMaskCreator([0, 1])),
        ([128, 128, 3, 3], BlockSparsityMaskCreator([2, 2])),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda availability")
def test_sparsity_mask_creator_cuda(tensor_shape, mask_creator, sparsity_val):
    _test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val, "cuda")


@pytest.mark.parametrize(
    ("creator_str,creator_obj"),
    [
        ("unstructured", UnstructuredSparsityMaskCreator()),
        ("channel", DimensionSparsityMaskCreator("channel")),
        ("filter", DimensionSparsityMaskCreator("filter")),
        ("[1, 4]", BlockSparsityMaskCreator([1, 4])),
        ([1, 4], BlockSparsityMaskCreator([1, 4])),
    ],
)
def test_mask_creator_serialization(creator_str, creator_obj):
    obj_from_creator_str = load_mask_creator(creator_str)
    str_from_creator_obj = str(creator_obj)
    print(str_from_creator_obj)
    assert obj_from_creator_str.__class__ == creator_obj.__class__
    assert str_from_creator_obj == str(creator_str)
    if "Dimension" in creator_str:
        assert obj_from_creator_str._dim == creator_obj._dim
    if "Block" in creator_str:
        assert obj_from_creator_str._block_shape == creator_obj._block_shape
