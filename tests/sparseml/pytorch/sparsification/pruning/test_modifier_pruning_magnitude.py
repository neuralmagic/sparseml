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
    GlobalMagnitudePruningModifier,
    GMPruningModifier,
    MagnitudePruningModifier,
)
from sparseml.pytorch.utils import tensor_sparsity
from sparseml.utils import FROM_PARAM_TOKEN
from tests.sparseml.pytorch.helpers import LinearNet
from tests.sparseml.pytorch.sparsification.pruning.helpers import (
    pruning_modifier_serialization_vals_test,
    state_dict_save_load_test,
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


@flaky(max_runs=3, min_passes=2)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: GMPruningModifier(
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=0.0,
            end_epoch=15.0,
            update_frequency=1.0,
            params=["re:.*weight"],
            inter_func="linear",
        ),
        lambda: GMPruningModifier(
            init_sparsity=FROM_PARAM_TOKEN,
            final_sparsity=0.95,
            start_epoch=0.0,
            end_epoch=15.0,
            update_frequency=1.0,
            params=["re:.*weight"],
            inter_func="linear",
        ),
        lambda: GlobalMagnitudePruningModifier(
            params="__ALL_PRUNABLE__",
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
        ),
        lambda: GMPruningModifier(
            params=["seq.fc1.weight", "seq.fc2.weight"],
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
            mask_type="block4",
        ),
        lambda: GMPruningModifier(
            params=["__ALL_PRUNABLE__"],
            init_sparsity=0.8,
            final_sparsity=0.9,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=2.0,
            inter_func="cubic",
        ),
        lambda: GMPruningModifier(
            params=[],
            init_sparsity=0.05,
            final_sparsity={
                0.6: ["seq.fc1.weight", "seq.fc2.weight"],
                0.8: ["re:seq.block1.*weight"],
            },
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
            mask_type="block4",
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
class TestGMPruningModifier(ScheduledUpdateModifierTest):
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

    def test_state_dict_save_load(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        state_dict_save_load_test(
            self,
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_steps_per_epoch,
            True,
        )


@pytest.mark.parametrize(
    "params,init_sparsity,final_sparsity",
    [
        (["re:.*weight"], 0.05, 0.8),
        (
            [],
            0.05,
            {0.7: ["param1"], 0.8: ["param2", "param3"], 0.9: ["param4", "param5"]},
        ),
        (["re:.*weight"], FROM_PARAM_TOKEN, 0.8),
        (
            [],
            FROM_PARAM_TOKEN,
            {0.7: ["param1"], 0.8: ["param2", "param3"], 0.9: ["param4", "param5"]},
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_gm_pruning_yaml(params, init_sparsity, final_sparsity):
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    inter_func = "cubic"
    mask_type = "filter"
    global_sparsity = False
    yaml_str = f"""
    !GMPruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
        global_sparsity: {global_sparsity}
    """
    yaml_modifier = GMPruningModifier.load_obj(yaml_str)  # type: GMPruningModifier
    serialized_modifier = GMPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: GMPruningModifier
    obj_modifier = GMPruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
    )

    assert isinstance(yaml_modifier, GMPruningModifier)
    pruning_modifier_serialization_vals_test(
        yaml_modifier, serialized_modifier, obj_modifier
    )

    assert (
        yaml_modifier.global_sparsity
        == serialized_modifier.global_sparsity
        == obj_modifier.global_sparsity
    )


def test_magnitude_pruning_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "block4"
    yaml_str = f"""
    !MagnitudePruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """
    yaml_modifier = MagnitudePruningModifier.load_obj(
        yaml_str
    )  # type: MagnitudePruningModifier
    serialized_modifier = MagnitudePruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: MagnitudePruningModifier
    obj_modifier = MagnitudePruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
    )

    assert isinstance(yaml_modifier, MagnitudePruningModifier)
    pruning_modifier_serialization_vals_test(
        yaml_modifier, serialized_modifier, obj_modifier
    )


def test_global_magnitude_pruning_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "filter"
    yaml_str = f"""
    !GlobalMagnitudePruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """
    yaml_modifier = GlobalMagnitudePruningModifier.load_obj(yaml_str)
    serialized_modifier = GlobalMagnitudePruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: GlobalMagnitudePruningModifier
    obj_modifier = GlobalMagnitudePruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
    )

    assert isinstance(yaml_modifier, GlobalMagnitudePruningModifier)
    pruning_modifier_serialization_vals_test(
        yaml_modifier, serialized_modifier, obj_modifier
    )
