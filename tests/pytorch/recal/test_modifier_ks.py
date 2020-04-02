import pytest

from torch.nn import Sequential, Linear, ReLU
from torch.optim import SGD

from neuralmagicML.pytorch.recal import GradualKSModifier, ConstantKSModifier

from tests.pytorch.recal.test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
    def_model,
)


TEST_MODEL = Sequential(
    Linear(8, 16), ReLU(), Sequential(Linear(16, 32), ReLU()), Linear(32, 1), ReLU()
)
TEST_MODEL_LAYER = "2.0"


CONSTANT_KS_MODIFIER = [
    lambda: ConstantKSModifier(layers="__ALL__",),
    lambda: ConstantKSModifier(
        layers=[TEST_MODEL_LAYER], start_epoch=10.0, end_epoch=25.0,
    ),
]


@pytest.mark.parametrize("modifier_lambda", CONSTANT_KS_MODIFIER, scope="function")
@pytest.mark.parametrize("model_lambda", [lambda: TEST_MODEL], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [lambda model: SGD(model.parameters(), 0.001)], scope="function",
)
class TestConstantKSModifier(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
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


GRADUAL_KS_MODIFIER = [
    lambda: GradualKSModifier(
        layers=[TEST_MODEL_LAYER],
        init_sparsity=0.05,
        final_sparsity=0.95,
        start_epoch=0.0,
        end_epoch=15.0,
        update_frequency=1.0,
        param="weight",
        inter_func="linear",
    ),
    lambda: GradualKSModifier(
        layers=[TEST_MODEL_LAYER],
        init_sparsity=0.05,
        final_sparsity=0.95,
        start_epoch=10.0,
        end_epoch=25.0,
        update_frequency=1.0,
        inter_func="cubic",
    ),
]


@pytest.mark.parametrize("modifier_lambda", GRADUAL_KS_MODIFIER, scope="function")
@pytest.mark.parametrize("model_lambda", [lambda: TEST_MODEL], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [lambda model: SGD(model.parameters(), 0.001)], scope="function",
)
class TestGradualKSModifier(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier, model, optimizer = self.create_test_objs(
            modifier_lambda, model_lambda, optim_lambda
        )
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
