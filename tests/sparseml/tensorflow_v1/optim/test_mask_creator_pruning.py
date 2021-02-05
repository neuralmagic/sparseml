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

import random

import pytest

from sparseml.tensorflow_v1.optim import (
    BlockPruningMaskCreator,
    DimensionPruningMaskCreator,
    GroupedPruningMaskCreator,
    UnstructuredPruningMaskCreator,
    load_mask_creator,
)
from sparseml.tensorflow_v1.utils import eval_tensor_sparsity, tf_compat


@pytest.mark.flaky
@pytest.mark.parametrize(
    ("tensor_shape,mask_creator"),
    [
        ([64, 64], UnstructuredPruningMaskCreator()),
        ([3, 3, 64, 64], UnstructuredPruningMaskCreator()),
        ([512, 64], DimensionPruningMaskCreator("channel")),
        ([3, 3, 512, 64], DimensionPruningMaskCreator("channel")),
        ([3, 3, 64, 64], DimensionPruningMaskCreator("filter")),
        ([512, 64], BlockPruningMaskCreator([4, 1])),
        ([3, 3, 512, 64], BlockPruningMaskCreator([4, 1])),
        ([3, 3, 128, 128], BlockPruningMaskCreator([2, 2])),
        ([3, 3, 128, 128], BlockPruningMaskCreator([-1, 1])),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_sparsity_mask_creator(tensor_shape, mask_creator, sparsity_val):
    tensor = tf_compat.Variable(tf_compat.random_normal(tensor_shape))
    sparsity = tf_compat.constant(sparsity_val)
    name = "{}_{}_{}_{}".format(
        mask_creator.__class__.__name__,
        "_".join([str(s) for s in tensor_shape]),
        str(sparsity_val),
        random.randint(0, 10000),
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

        eval_tensor_sparsity(initial_mask, sess)
        new_mask_sparsity = eval_tensor_sparsity(new_mask, sess)

        assert abs(new_mask_sparsity - sparsity_val) < 1e-2
        if isinstance(mask_creator, GroupedPruningMaskCreator):
            # Check that every value in the mask_creator grouping
            # is the same within the mask.  Assumes grouping applies
            # an absolte mean to each grouping
            for mask in [initial_mask, new_mask]:
                grouped_mask = mask_creator.group_tensor(mask)
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
        ("unstructured", UnstructuredPruningMaskCreator()),
        ("channel", DimensionPruningMaskCreator("channel")),
        ("filter", DimensionPruningMaskCreator("filter")),
        ("[4, 1]", BlockPruningMaskCreator([4, 1])),
        ([4, 1], BlockPruningMaskCreator([4, 1])),
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
