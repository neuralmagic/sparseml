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

import math
import os

import pytest
import torch

from flaky import flaky
from sparseml.pytorch.nn import Identity
from sparseml.pytorch.optim import (
    ConstantPruningModifier,
    GlobalMagnitudePruningModifier,
    GMPruningModifier,
    LayerPruningModifier,
    MagnitudePruningModifier,
    MFACGlobalPruningModifier,
    MFACPruningModifier,
    MovementPruningModifier,
    StructuredPruningModifier,
    load_mask_creator,
)
from sparseml.pytorch.utils import get_layer
from tests.sparseml.pytorch.helpers import FlatMLPNet, LinearNet
from tests.sparseml.pytorch.optim.test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    create_optim_adam,
    create_optim_sgd,
)


from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)


def _test_state_dict_save_load(
    test_obj,
    modifier_lambda,
    model_lambda,
    optim_lambda,
    test_steps_per_epoch,  # noqa: F811
    is_gm_pruning,
):
    modifier = modifier_lambda()
    model = model_lambda()
    optimizer = optim_lambda(model)
    test_obj.initialize_helper(modifier, model)
    # apply first mask
    modifier.scheduled_update(
        model, optimizer, modifier.start_epoch, test_steps_per_epoch
    )
    # get state dict
    state_dict = modifier.state_dict()
    applied_sparsities = modifier.applied_sparsity if is_gm_pruning else None
    if not isinstance(applied_sparsities, list):
        applied_sparsities = [applied_sparsities] * len(state_dict)
    for mask, applied_sparsity in zip(state_dict.values(), applied_sparsities):
        if is_gm_pruning:
            # check that the mask sparsity is the applied one leaving a relatively
            # large margin of error since parameter sizes are small so the exact
            # sparsity cannot always be attained
            assert abs(1 - (mask.sum() / mask.numel()) - applied_sparsity) < 0.05
        else:
            # all weights should be non zero, pending randomness, so the mask should be
            # all ones for this constant_pruning modifier
            assert mask.sum() / mask.numel() >= 0.99

    # check that changing the state dict masks to all 0s and reapplying will affect
    # the model parameters
    for mask in state_dict.values():
        mask.mul_(0.0)
    modifier.load_state_dict(state_dict)
    param_names = {mask_name.split(".sparsity_mask")[0] for mask_name in state_dict}
    for param_name, param in model.named_parameters():
        if param_name in param_names:
            # check that the all zero mask has been applied
            assert torch.all(param == 0.0)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: ConstantPruningModifier(
            params=["re:.*weight"],
        ),
        lambda: ConstantPruningModifier(
            params=["seq.fc1.weight"],
            start_epoch=10.0,
            end_epoch=25.0,
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
class TestConstantPruningModifier(ScheduledModifierTest):
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

        # check sparsity is not set before
        if modifier.start_epoch >= 0:
            for epoch in range(int(modifier.start_epoch)):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

        epoch = int(modifier.start_epoch) if modifier.start_epoch >= 0 else 0.0
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        if modifier.end_epoch >= 0:
            epoch = int(modifier.end_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            for epoch in range(
                int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6
            ):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

    def test_state_dict_save_load(
        self,
        modifier_lambda,
        model_lambda,
        optim_lambda,
        test_steps_per_epoch,  # noqa: F811
    ):
        _test_state_dict_save_load(
            self,
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_steps_per_epoch,
            False,
        )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_constant_pruning_yaml():
    start_epoch = 5.0
    end_epoch = 15.0
    params = ["re:.*weight"]
    yaml_str = f"""
    !ConstantPruningModifier
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        params: {params}
    """
    yaml_modifier = ConstantPruningModifier.load_obj(
        yaml_str
    )  # type: ConstantPruningModifier
    serialized_modifier = ConstantPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: ConstantPruningModifier
    obj_modifier = ConstantPruningModifier(
        start_epoch=start_epoch, end_epoch=end_epoch, params=params
    )

    assert isinstance(yaml_modifier, ConstantPruningModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params


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
            params=["re:seq.block1.*weight"],
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
            global_sparsity=True,
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
            mask_type=[1, 4],
        ),
        lambda: GMPruningModifier(
            params=["__ALL_PRUNABLE__"],
            init_sparsity=0.9,
            final_sparsity=0.9,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=2.0,
            inter_func="cubic",
            phased=True,
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
            mask_type=[1, 4],
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
        assert type(load_mask_creator(modifier._mask_type)) == type(  # noqa: E721
            modifier._mask_creator
        )
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
        assert all(
            applied_sparsity == modifier.init_sparsity
            for applied_sparsity in applied_sparsities
        )

        last_sparsities = [modifier.init_sparsity]

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

            if not modifier.phased:
                assert all(
                    applied_sparsity > last_sparsity
                    for applied_sparsity, last_sparsity in zip(
                        applied_sparsities, last_sparsities
                    )
                )
            else:
                pruned_on = (
                    math.floor(
                        (epoch - modifier.start_epoch) / modifier.update_frequency
                    )
                    % 2
                    == 0
                )
                assert all(
                    (
                        (applied_sparsity > last_sparsity)
                        if pruned_on
                        else (applied_sparsity == 0)
                    )
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
            if isinstance(modifier.final_sparsity, float):
                assert modifier.applied_sparsity == modifier.final_sparsity
            else:
                assert all(
                    sparsity in modifier.final_sparsity
                    for sparsity in modifier.applied_sparsity
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
        _test_state_dict_save_load(
            self,
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_steps_per_epoch,
            True,
        )


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: LayerPruningModifier(
            layers=["seq.fc2", "seq.act2"],
            start_epoch=0.0,
            end_epoch=15.0,
            update_frequency=1.0,
        ),
        lambda: LayerPruningModifier(
            layers=["seq.fc2", "seq.act2"],
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
        ),
        lambda: LayerPruningModifier(
            layers="__ALL_PRUNABLE__",
            start_epoch=10.0,
            end_epoch=25.0,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [FlatMLPNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda",
    [create_optim_sgd, create_optim_adam],
    scope="function",
)
class TestLayerPruningModifier(ScheduledUpdateModifierTest):
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
        assert len(modifier._layer_modules) > 0
        if modifier.start_epoch > 0:
            for (name, mod) in modifier._layer_modules.items():
                assert mod is None
                assert not isinstance(get_layer(name, model), Identity)

            # check sparsity is not set before
            for epoch in range(int(modifier.start_epoch)):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

            for (name, mod) in modifier._layer_modules.items():
                assert mod is None
                assert not isinstance(get_layer(name, model), Identity)

            epoch = int(modifier.start_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        else:
            epoch = 0

        for (name, mod) in modifier._layer_modules.items():
            assert mod is not None
            assert isinstance(get_layer(name, model), Identity)

        # check forward pass
        input_shape = model_lambda.layer_descs()[0].input_size
        test_batch = torch.randn(10, *input_shape)
        _ = model(test_batch)

        end_epoch = (
            modifier.end_epoch if modifier.end_epoch > -1 else modifier.start_epoch + 10
        )

        while epoch < end_epoch - 0.1:
            epoch += 0.1
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        _ = model(test_batch)  # check forward pass

        if modifier.end_epoch > -1:
            epoch = int(modifier.end_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            for (name, mod) in modifier._layer_modules.items():
                assert mod is None
                assert not isinstance(get_layer(name, model), Identity)


def _test_pruning_modifier_serialization_vals(
    yaml_modifier, serialized_modifier, obj_modifier
):
    assert (
        yaml_modifier.init_sparsity
        == serialized_modifier.init_sparsity
        == obj_modifier.init_sparsity
    )
    assert (
        yaml_modifier.final_sparsity
        == serialized_modifier.final_sparsity
        == obj_modifier.final_sparsity
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
    assert (
        str(yaml_modifier.mask_type)
        == str(serialized_modifier.mask_type)
        == str(obj_modifier.mask_type)
    )


@pytest.mark.parametrize(
    "params, final_sparsity",
    [
        (["re:.*weight"], 0.8),
        ([], {0.7: ["param1"], 0.8: ["param2", "param3"], 0.9: ["param4", "param5"]}),
    ],
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_gm_pruning_yaml(params, final_sparsity):
    init_sparsity = 0.05
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
    _test_pruning_modifier_serialization_vals(
        yaml_modifier, serialized_modifier, obj_modifier
    )
    assert (
        str(yaml_modifier.global_sparsity)
        == str(serialized_modifier.global_sparsity)
        == str(obj_modifier.global_sparsity)
    )


def test_magnitude_pruning_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "filter"
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
    serialized_modifier = GMPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: MagnitudePruningModifier
    obj_modifier = GMPruningModifier(
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
    _test_pruning_modifier_serialization_vals(
        yaml_modifier, serialized_modifier, obj_modifier
    )


def test_movement_pruning_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "filter"
    yaml_str = f"""
    !MovementPruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """
    yaml_modifier = MovementPruningModifier.load_obj(
        yaml_str
    )  # type: MovementPruningModifier
    serialized_modifier = MovementPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: MovementPruningModifier
    obj_modifier = MovementPruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
    )

    assert isinstance(yaml_modifier, MovementPruningModifier)
    _test_pruning_modifier_serialization_vals(
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
    _test_pruning_modifier_serialization_vals(
        yaml_modifier, serialized_modifier, obj_modifier
    )


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
    _test_pruning_modifier_serialization_vals(
        yaml_modifier, serialized_modifier, obj_modifier
    )
    assert (
        yaml_modifier.param_groups
        == serialized_modifier.param_groups
        == obj_modifier.param_groups
    )


def test_mfac_pruning_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "unstructured"
    global_sparsity = False
    mfac_options = {"num_grads": 64, "available_gpus": ["cuda:0", "cuda:1"]}
    yaml_str = f"""
    !MFACPruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
        global_sparsity: {global_sparsity}
        mfac_options: {mfac_options}
    """
    yaml_modifier = MFACPruningModifier.load_obj(yaml_str)
    serialized_modifier = MFACPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: MFACPruningModifier
    obj_modifier = MFACPruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
        global_sparsity=global_sparsity,
        mfac_options=mfac_options,
    )

    assert isinstance(yaml_modifier, MFACPruningModifier)
    _test_pruning_modifier_serialization_vals(
        yaml_modifier, serialized_modifier, obj_modifier
    )
    assert (
        str(yaml_modifier.global_sparsity)
        == str(serialized_modifier.global_sparsity)
        == str(obj_modifier.global_sparsity)
    )
    assert (
        yaml_modifier.mfac_options
        == serialized_modifier.mfac_options
        == obj_modifier.mfac_options
    )


def test_global_mfac_pruning_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = "__ALL_PRUNABLE__"
    inter_func = "cubic"
    mask_type = "unstructured"
    mfac_options = {"num_grads": 64, "available_gpus": ["cuda:0", "cuda:1"]}
    yaml_str = f"""
    !MFACGlobalPruningModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
        mfac_options: {mfac_options}
    """
    yaml_modifier = MFACGlobalPruningModifier.load_obj(yaml_str)
    serialized_modifier = MFACGlobalPruningModifier.load_obj(
        str(yaml_modifier)
    )  # type: MFACGlobalPruningModifier
    obj_modifier = MFACGlobalPruningModifier(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
        mfac_options=mfac_options,
    )

    assert isinstance(yaml_modifier, MFACGlobalPruningModifier)
    _test_pruning_modifier_serialization_vals(
        yaml_modifier, serialized_modifier, obj_modifier
    )
    assert (
        yaml_modifier.mfac_options
        == serialized_modifier.mfac_options
        == obj_modifier.mfac_options
    )
