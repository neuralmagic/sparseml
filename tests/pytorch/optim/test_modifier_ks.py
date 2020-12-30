import pytest
import os

import torch
from torch.optim import SGD

from sparseml.pytorch.optim import (
    GMPruningModifier,
    ConstantPruningModifier,
    DimensionSparsityMaskCreator,
    BlockPruningMaskCreator,
)

from tests.pytorch.helpers import LinearNet
from tests.pytorch.optim.test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
    create_optim_sgd,
    create_optim_adam,
)


def _test_state_dict_save_load(
    test_obj,
    modifier_lambda,
    model_lambda,
    optim_lambda,
    test_steps_per_epoch,
    is_gradual_ks,
):
    modifier = modifier_lambda()
    model = model_lambda()
    optimizer = optim_lambda(model)
    test_obj.initialize_helper(modifier, model, optimizer)
    # apply first mask
    modifier.scheduled_update(
        model, optimizer, modifier.start_epoch, test_steps_per_epoch
    )
    # get state dict
    state_dict = modifier.state_dict()
    for mask in state_dict.values():
        if is_gradual_ks:
            # check that the mask sparsity is the applied one leaving a relatively
            # large margin of error since parameter sizes are small so the exact sparsity
            # cannot always be attained
            assert (
                abs(1 - (mask.sum() / mask.numel()) - modifier.applied_sparsity) < 0.05
            )
        else:
            # all weights should be non zero, pending randomness, so the mask should be
            # all ones for this constant_ks modifier
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: ConstantPruningModifier(params=["re:.*weight"], ),
        lambda: ConstantPruningModifier(
            params=["seq.fc1.weight"], start_epoch=10.0, end_epoch=25.0,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function",
)
class TestConstantKSModifier(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)

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
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
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
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_constant_ks_yaml():
    start_epoch = 5.0
    end_epoch = 15.0
    params = ["re:.*weight"]
    yaml_str = """
    !ConstantKSModifier
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        params: {params}
    """.format(
        start_epoch=start_epoch, end_epoch=end_epoch, params=params
    )
    yaml_modifier = ConstantPruningModifier.load_obj(yaml_str)  # type: ConstantPruningModifier
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


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
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
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function",
)
class TestGradualKSModifier(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)
        assert modifier.applied_sparsity is None

        # check sparsity is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity is None

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert modifier.applied_sparsity == modifier.init_sparsity
        last_sparsity = modifier.init_sparsity

        while epoch < modifier.end_epoch - modifier.update_frequency:
            epoch += modifier.update_frequency
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity > last_sparsity
            last_sparsity = modifier.applied_sparsity

        epoch = int(modifier.end_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert modifier.applied_sparsity == modifier.final_sparsity

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity == modifier.final_sparsity

    def test_state_dict_save_load(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        _test_state_dict_save_load(
            self,
            modifier_lambda,
            model_lambda,
            optim_lambda,
            test_steps_per_epoch,
            True,
        )


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_gradual_ks_yaml():
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    params = ["re:.*weight"]
    inter_func = "cubic"
    mask_type = "filter"
    yaml_str = """
    !GradualKSModifier
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        params: {params}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """.format(
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        params=params,
        inter_func=inter_func,
        mask_type=mask_type,
    )
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
    )

    assert isinstance(yaml_modifier, GMPruningModifier)
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
