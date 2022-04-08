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
from sparseml.pytorch.sparsification import (
    StructuredPruningMaskCreator,
    StructuredPruningModifier,
)
from sparseml.pytorch.utils import tensor_sparsity
from tests.sparseml.pytorch.helpers import LinearNet
from tests.sparseml.pytorch.sparsification.pruning.helpers import (
    pruning_modifier_serialization_vals_test,
    sparsity_mask_creator_test,
)
from tests.sparseml.pytorch.sparsification.test_modifier import (
    ScheduledUpdateModifierTest,
    create_optim_adam,
    create_optim_sgd,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    ("tensor_shape,structure_type,tensor_group_idxs"),
    [
        ([[64, 64]] * 10, "filter", None),
        ([[64, 64, 3, 3], [64, 64]], "channel", None),
        ([[64, 64]] * 6, "filter", [[0, 3], [1, 2, 5], [4]]),
        ([[64, 64, 3, 3]] * 4, "channel", [[0, 1, 2, 3]]),
    ],
)
@pytest.mark.parametrize("sparsity_val", [0.0, 0.4, 0.6, 0.9, 0.99, 1.0])
def test_structured_sparsity_mask_creator(
    tensor_shape, structure_type, tensor_group_idxs, sparsity_val
):
    mask_creator = StructuredPruningMaskCreator(
        structure_type, tensor_group_idxs=tensor_group_idxs
    )
    masks = sparsity_mask_creator_test(tensor_shape, mask_creator, sparsity_val, "cpu")
    target_dim = 1 if structure_type == "filter" else 0
    for mask in masks:
        dimension_mask = 1 - torch.all(mask == 0.0, dim=target_dim).float()
        assert abs(tensor_sparsity(dimension_mask) - sparsity_val) <= 0.01
    if tensor_group_idxs:
        for group in tensor_group_idxs:
            first_mask = masks[group[0]]
            for mask_idx in group:
                assert torch.all(masks[mask_idx] == first_mask)


@flaky(max_runs=3, min_passes=2)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: StructuredPruningModifier(
            params="__ALL_PRUNABLE__",
            param_groups=[["seq.fc1.weight", "seq.fc2.weight"]],
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [create_optim_sgd, create_optim_adam],
    scope="function",
)
class TestStructuredPruningModifier(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model)

        # test tensor_group_idxs correctness
        tensor_group_idxs = modifier.mask_creator.tensor_group_idxs
        param_names_full = [
            f"{layer_name}.{param_name}"
            for layer_name, param_name in zip(
                modifier.module_masks.layer_names, modifier.module_masks.param_names
            )
        ]
        param_names_to_idx = dict(zip(param_names_full, range(len(param_names_full))))
        expected_tensor_group_idxs = [
            {param_names_to_idx[param_name] for param_name in param_group}
            for param_group in modifier.param_groups
            if len(param_group) > 1
        ]
        found_tensor_group_idxs = [
            set(tensor_group)
            for tensor_group in tensor_group_idxs
            if len(tensor_group) > 1
        ]
        assert len(expected_tensor_group_idxs) == len(found_tensor_group_idxs)
        for expected_tensor_group in expected_tensor_group_idxs:
            assert any(
                tensor_group == expected_tensor_group
                for tensor_group in found_tensor_group_idxs
            )

        if modifier.start_epoch > 0:
            assert modifier.applied_sparsity is None
        assert modifier._mask_creator == modifier._module_masks._mask_creator

        # check sparsity is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity is None

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        applied_sparsities = modifier.applied_sparsity
        if not isinstance(applied_sparsities, list):
            applied_sparsities = [applied_sparsities]

        if not isinstance(modifier.init_sparsity, str):
            assert all(
                applied_sparsity == modifier.init_sparsity
                for applied_sparsity in applied_sparsities
            )
        else:
            assert len(modifier._init_sparsity) == len(modifier.module_masks.layers)
            for idx, param in enumerate(modifier.module_masks.params_data):
                assert modifier._init_sparsity[idx] == tensor_sparsity(param).item()

        last_sparsities = applied_sparsities

        # check forward pass
        input_shape = model_lambda.layer_descs()[0].input_size
        test_batch = torch.randn(10, *input_shape)
        _ = model(test_batch)

        while epoch < modifier.end_epoch - modifier.update_frequency:
            epoch += modifier.update_frequency
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            applied_sparsities = modifier.applied_sparsity
            if not isinstance(applied_sparsities, list):
                applied_sparsities = [applied_sparsities]

            assert all(
                applied_sparsity > last_sparsity
                for applied_sparsity, last_sparsity in zip(
                    applied_sparsities, last_sparsities
                )
            )

            last_sparsities = applied_sparsities

        _ = model(test_batch)  # check forward pass
        epoch = int(modifier.end_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        def _test_final_sparsity_applied():
            final_sparsities = (
                [modifier.final_sparsity]
                if isinstance(modifier.final_sparsity, float)
                else modifier.final_sparsity
            )
            assert all(
                sparsity in final_sparsities for sparsity in modifier.applied_sparsity
            )

        _test_final_sparsity_applied()

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            _test_final_sparsity_applied()


def test_structured_pruning_yaml():
    param_groups = [
        ["param1", "param2"],
        [
            "param3",
            "param4",
            "param5",
        ],
    ]
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "filter"
    yaml_str = f"""
    !StructuredPruningModifier
        param_groups: {param_groups}
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """
    yaml_modifier = StructuredPruningModifier.load_obj(yaml_str)
    serialized_modifier = StructuredPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: StructuredPruningModifier
    obj_modifier = StructuredPruningModifier(
        param_groups=param_groups,
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
    )

    assert isinstance(yaml_modifier, StructuredPruningModifier)
    pruning_modifier_serialization_vals_test(
        yaml_modifier, serialized_modifier, obj_modifier
    )
    assert (
        yaml_modifier.param_groups
        == serialized_modifier.param_groups
        == obj_modifier.param_groups
    )
