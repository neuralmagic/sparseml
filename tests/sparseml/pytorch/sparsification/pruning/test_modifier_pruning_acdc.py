import pytest
import torch
from sparseml.pytorch.sparsification.pruning import ACDCPruningModifier
from torch.nn import Module
from torch.optim import SGD

from tests.sparseml.pytorch.helpers import LinearNet
from tests.sparseml.pytorch.helpers import (  # noqa isort:skip
    test_epoch,
    test_loss,
    test_steps_per_epoch,
)
from tests.sparseml.pytorch.optim.test_modifier import (
    ScheduledModifierTest,
    create_optim_adam,
)


def create_optim_sgd(model: Module, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0) -> SGD:
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)



@pytest.mark.parametrize("start_epoch,end_epoch_orig,update_frequency, expected_end_epoch",
                         [
                             (5, 15, 3, 11),
                             (0, 9, 2, 8),
                             (10, 20, 5, 20)
                         ])
def test_finish_on_compression(start_epoch, end_epoch_orig, update_frequency, expected_end_epoch):
    _, end_epoch = ACDCPruningModifier._compute_compression_phases(start_epoch, end_epoch_orig, update_frequency)
    assert end_epoch == expected_end_epoch



@pytest.mark.parametrize(
    "modifier_lambda",
    [lambda: ACDCPruningModifier(
        compression_sparsity=0.9,
        start_epoch=0,
        end_epoch=20,
        update_frequency=5,
        params=["re:.*weight"],
        momentum_buffer_reset=True
    ),
     lambda: ACDCPruningModifier(
         compression_sparsity=0.9,
         start_epoch=2,
         end_epoch=22,
         update_frequency=3,
         params=["re:.*weight"],
         momentum_buffer_reset=False
     ),
     lambda: ACDCPruningModifier(
         compression_sparsity=0.8,
         start_epoch=6,
         end_epoch=26,
         update_frequency=1,
         params=["re:.*weight"],
         momentum_buffer_reset=True
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
class TestACDCPruningModifier(ScheduledModifierTest):
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

        # assert that until modifier is activated, `applied_sparsity` remains None
        if modifier.start_epoch > 0:
            assert modifier.applied_sparsity is None
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity is None

        # assert that once `start_epoch` happens, modifier is ready for update.
        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        # check whether compression and decompression phases alternate properly
        is_previous_phase_decompression = None
        assert modifier._is_phase_decompression is True
        while epoch < modifier.end_epoch - modifier.update_frequency:
            epoch += modifier.update_frequency
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            if is_previous_phase_decompression is not None:
                assert is_previous_phase_decompression is not modifier._is_phase_decompression
            is_previous_phase_decompression = modifier._is_phase_decompression

        def _test_compression_sparsity_applied():
            assert modifier._compression_sparsity == modifier.applied_sparsity

        _test_compression_sparsity_applied()

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6):
        #    assert not modifier.update_ready(epoch, test_steps_per_epoch)
            _test_compression_sparsity_applied()

    def test_momentum_mechanism(
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

        batch_shape = 10
        input_shape = model_lambda.layer_descs()[0].input_size
        epoch = int(modifier.start_epoch)
        is_previous_phase_decompression = None
        while epoch < modifier.end_epoch:
            optimizer.zero_grad()
            model(torch.randn(batch_shape, *input_shape)).mean().backward()
            optimizer.step()
            if modifier.update_ready(epoch, test_steps_per_epoch):
                modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            modifier.optimizer_post_step(model, optimizer, epoch, test_steps_per_epoch)
            if modifier._momentum_buffer_reset and isinstance(optimizer, SGD):
                for name, param in model.named_parameters():
                    momentum_buffer = optimizer.state[param]['momentum_buffer']

                    if is_previous_phase_decompression is not None \
                            and not is_previous_phase_decompression \
                            and modifier._is_phase_decompression:
                        assert torch.all(momentum_buffer == 0.0).item() is True

                    else:
                        assert torch.all(momentum_buffer == 0.0).item() is False

            is_previous_phase_decompression = modifier._is_phase_decompression
            epoch += 1



