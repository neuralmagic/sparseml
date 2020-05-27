import pytest
import random

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    eval_tensor_sparsity,
)
from neuralmagicML.tensorflow.recal import (
    UnstructuredSparsityMaskCreator,
    GroupedSparsityMaskCreator,
    DimensionSparsityMaskCreator,
    BlockSparsityMaskCreator,
    load_mask_creator,
)


@pytest.mark.parametrize(
    ("tensor_shape,mask_creator"),
    [
        ([64, 64], UnstructuredSparsityMaskCreator()),
        ([3, 3, 64, 64], UnstructuredSparsityMaskCreator()),
        ([512, 64], DimensionSparsityMaskCreator('channel')),
        ([3, 3, 512, 64], DimensionSparsityMaskCreator('channel')),
        ([3, 3, 64, 64], DimensionSparsityMaskCreator('filter')),
        ([512, 64], BlockSparsityMaskCreator([4, 1])),
        ([3, 3, 512, 64], BlockSparsityMaskCreator([4, 1])),
        ([3, 3, 128, 128], BlockSparsityMaskCreator([2, 2])),
        ([3, 3, 128, 128], BlockSparsityMaskCreator([-1, 1])),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val):
    tensor = tf_compat.Variable(tf_compat.random_normal(tensor_shape))
    sparsity = tf_compat.constant(sparsity_val)
    name = (
        f"{mask_creator.__class__.__name__}_{'_'.join([str(s) for s in tensor_shape])}"
        f"_{str(sparsity_val)}_{random.randint(0, 10000)}"
    )
    initial_mask = tf_compat.get_variable(
        name,
        tensor_shape,
        initializer=mask_creator.get_mask_initializer(tensor),
        dtype=tensor.dtype,
    )
    new_mask = mask_creator.create_sparsity_mask(tensor, sparsity)
    assert initial_mask.shape == tensor.shape
    assert new_mask.shape == tensor.shape
    with tf_compat.Session() as sess:
        sess.run(tf_compat.global_variables_initializer())

        initial_mask_sparsity = eval_tensor_sparsity(initial_mask, sess)
        new_mask_sparsity = eval_tensor_sparsity(new_mask, sess)

        assert abs(new_mask_sparsity - sparsity_val) < 1e-2
        if isinstance(mask_creator, GroupedSparsityMaskCreator):
            # Check that every value in the mask_creator grouping
            # is the same within the mask.  Assumes grouping applies
            # an absolte mean to each grouping
            for mask in [initial_mask, new_mask]:
                grouped_mask = mask_creator._group_tensor(mask)
                mask_vals_are_grouped = tf_compat.reduce_all(
                    tf_compat.logical_or(
                        tf_compat.equal(grouped_mask, 0.0),
                        tf_compat.equal(grouped_mask, 1.0),
                    )
                )
                assert sess.run(mask_vals_are_grouped)


@pytest.mark.parametrize(
    ("creator_str,creator_obj"),
    [
        ("unstructured", UnstructuredSparsityMaskCreator()),
        ("channel", DimensionSparsityMaskCreator("channel")),
        ("filter", DimensionSparsityMaskCreator("filter")),
        ("[4, 1]", BlockSparsityMaskCreator([4, 1])),
        ([4, 1], BlockSparsityMaskCreator([4, 1])),
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
